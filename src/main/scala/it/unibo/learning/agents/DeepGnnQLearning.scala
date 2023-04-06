package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalLearner
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, DecayReference, GraphReplayBuffer}
import it.unibo.learning.network.torch._
import it.unibo.learning.network.torch.{util => torchUtil}
import it.unibo.learning.network.{Graph, GraphNeuralNetworkRL}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.util.Random

class DeepGnnQLearning(
    epsilon: DecayReference[Double],
    alpha: Double,
    gamma: Double,
    copyEach: Int,
    referenceNet: GraphNeuralNetworkRL,
    deviceName: String = "cuda:0"
) extends GraphLearner {
  private val device =
    torch.device(if (deviceName != "cpu" && torch.cuda.is_available().as[Boolean]) deviceName else "cpu")
  var updates = 0
  private var random: Random = new Random()
  private val targetNetwork = referenceNet.cloneNetwork
  private val policyNetwork = referenceNet
  targetNetwork.underlying.to(device)
  policyNetwork.underlying.to(device)
  private val optimizer = optim.RMSprop(policyNetwork.underlying.parameters(), alpha)
  private val encoder = targetNetwork.encoder
  private val behaviouralPolicy = policyNetwork.policy(device)
  targetNetwork.underlying.eval()
  policyNetwork.underlying.eval()

  override def policy: Graph[AgentState] => Graph[Int] = state => {
    if (random.nextDouble() < epsilon.value) {
      val action = state.data.map(_ => referenceNet.actionSpace.indices(random.nextInt(referenceNet.actionSpace.size)))
      Graph[Int](action, state.connections)
    } else {
      behaviouralPolicy(state)
    }
  }

  override def store(where: String): Unit = {}

  override def load(where: String): Graph[AgentState] => Graph[Int] = ???

  implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()
  import session._
  override def update(batch: Seq[GraphReplayBuffer.GraphExperience]): Unit = {

    // context
    // targetNetwork.underlying.train()
    // policyNetwork.underlying.train()
    //val states = referenceNet.encoder.encodeBatchNormalize(batch.map(_.stateT).map(referenceNet.encoder.encode(_, device = device).record()))
    //states.record()
    val states = encoder.encodeBatchNormalize[Seq[Double]](batch.map(_.stateT).map(encoder.encode(_, device = device).record()), device)(_.toPythonCopy).record()
    val action = referenceNet.encoder.encodeBatch[Int](batch.map(_.actionT).map(referenceNet.encoder.encodeInt(_, device).record()), device)(identity[Int]).record()
    val rewardsRaw = referenceNet.encoder.encodeBatch[Double](
      batch.map(_.rewardTPlus).map(referenceNet.encoder.encodeDouble(_, device).record()), device)(identity[Double]
    ).record().x.record()
    val rewards = torch.nn.functional.normalize(rewardsRaw, dim = 0).record().to(device).record()

    val nextStates = encoder.encodeBatchNormalize[Seq[Double]](batch.map(_.stateTPlus).map(referenceNet.encoder.encode(_, device = device).record()), device)(_.toPythonCopy).record()

    val stateActionValue =
      policyNetwork
        .forward(states)
        .record()
        .gather(
          1,
          action.x.to(device)
            .record()
            .view(-1, 1)
            .record()
        )
        .record()
    val nextStateValues = py.`with`(torch.no_grad()) { _ =>
      targetNetwork
        .forward(nextStates)
        .record()
        .max(1)
        .record()
        .bracketAccess(0)
        .record()
    }

    val expectedValue = ((nextStateValues * gamma).record() + rewards).record()
    val criterion = nn.SmoothL1Loss()
    val loss = criterion(stateActionValue, expectedValue.unsqueeze(1).record()).record()
    optimizer.zero_grad()
    loss.backward().record()
    writer.add_scalar("Loss", loss.detach().item().as[Double], updates)
    torch.nn.utils.clip_grad_value_(policyNetwork.underlying.parameters(), 1)
    optimizer.step()

    updates += 1
    if (updates % copyEach == 0) {
      targetNetwork.underlying.load_state_dict(policyNetwork.underlying.state_dict())
    }
    session.clear()
    //println(torchUtil.getAllTensors().size)
    // targetNetwork.underlying.eval()
    // policyNetwork.underlying.eval()
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: AbstractGlobalLearner): Unit = agent.attachDecayable("epsilon" -> epsilon)
}
