package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalLearner
import it.unibo.learning.abstractions.{AgentState, DecayReference, GraphReplayBuffer}
import it.unibo.learning.network.torch._
import it.unibo.learning.network.{Graph, GraphNeuralNetworkRL}
import me.shadaj.scalapy.py

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

  override def update(batch: Seq[GraphReplayBuffer.GraphExperience]): Unit = {
    implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()
    // context
    import session._
    // targetNetwork.underlying.train()
    // policyNetwork.underlying.train()
    val states = referenceNet.encoder.encodeBatchNormalize(batch.map(_.stateT).map(referenceNet.encoder.encode(_, device = device).record()))
    val action = referenceNet.encoder.encodeBatch(batch.map(_.actionT).map(referenceNet.encoder.encodeInt(_, device).record()))
    val rewards = torch.nn.functional
      .normalize(
        referenceNet.encoder.encodeBatch(batch.map(_.rewardTPlus).map(referenceNet.encoder.encodeDouble(_, device).record())).record().x,
        dim = 0
      ).record().to(device)
      .record()
    val nextStates = referenceNet.encoder.encodeBatchNormalize(batch.map(_.stateTPlus).map(referenceNet.encoder.encode(_, device = device).record())).record()
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


    referenceNet.underlying.applyDynamic()
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
    session.clear()
    updates += 1
    if (updates % copyEach == 0) {
      targetNetwork.underlying.load_state_dict(policyNetwork.underlying.state_dict())
    }
    // targetNetwork.underlying.eval()
    // policyNetwork.underlying.eval()
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: AbstractGlobalLearner): Unit = agent.attachDecayable("epsilon" -> epsilon)
}
