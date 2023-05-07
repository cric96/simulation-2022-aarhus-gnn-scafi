package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.DecayableSource
import it.unibo.learning.abstractions.{AgentState, DecayReference}
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.network.{Graph, GraphNeuralNetworkRL, NeuralNetworkRL}
import it.unibo.learning.network.torch.{PythonMemoryManager, nn, optim, torch, writer}
import me.shadaj.scalapy.py

import scala.util.Random

abstract class AbstractDeepQLearning[F[_], N <: NeuralNetworkRL[F, N]]() extends Learner[F] {
  def epsilon: DecayReference[Double]
  def alpha: Double
  def gamma: Double
  def copyEach: Int
  def referenceNet: N
  def deviceName: String = "cpu"
  protected val device =
    torch.device(if (deviceName != "cpu" && torch.cuda.is_available().as[Boolean]) deviceName else "cpu")
  var updates = 0
  protected var random: Random = new Random()
  protected val targetNetwork: N = referenceNet.cloneNetwork
  protected val policyNetwork: N = referenceNet
  targetNetwork.underlying.to(device)
  policyNetwork.underlying.to(device)
  protected val optimizer = optim.RMSprop(policyNetwork.underlying.parameters(), alpha)
  protected val encoder = targetNetwork.encoder
  protected val behaviouralPolicy = policyNetwork.policy(device)
  targetNetwork.underlying.eval()
  policyNetwork.underlying.eval()

  implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()

  import session._

  override def store(where: String): Unit = torch.save(targetNetwork.underlying.state_dict(), where)

  override def load(where: String): Unit = {
    targetNetwork.underlying.load_state_dict(torch.load(where))
    policyNetwork.underlying.load_state_dict(torch.load(where))
  }

  override def update(batch: Seq[Experience[F]]): Unit = {
    targetNetwork.underlying.train()
    policyNetwork.underlying.train()
    val states = encodeStates(batch.map(_.stateT)).record()
    val action = encodeActions(batch.map(_.actionT)).record()
    val rewards = encodeRewards(batch.map(_.rewardTPlus)).record()
    val nextStates = encodeStates(batch.map(_.stateTPlus)).record()
    val stateActionValue =
      policyNetwork
        .forward(states)
        .record()
        .gather(
          1,
          action
            .to(device)
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
    targetNetwork.underlying.eval()
    policyNetwork.underlying.eval()
  }

  def encodeStates(states: Seq[F[AgentState]]): py.Dynamic

  def encodeActions(actions: Seq[F[Int]]): py.Dynamic

  def encodeRewards(rewards: Seq[F[Double]]): py.Dynamic

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: DecayableSource): Unit = agent.attachDecayable("epsilon" -> epsilon)
}
