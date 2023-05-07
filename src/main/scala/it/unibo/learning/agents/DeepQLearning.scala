package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.DecayableSource
import it.unibo.learning.abstractions.{AgentState, DecayReference, ReplayBuffer}
import it.unibo.learning.network.MlpNeuralNetworkRL
import it.unibo.learning.network.torch._
import it.unibo.typelevel.Id
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.util.Random

class DeepQLearning(
    override val epsilon: DecayReference[Double],
    override val alpha: Double,
    override val gamma: Double,
    override val copyEach: Int,
    override val referenceNet: MlpNeuralNetworkRL,
    override val deviceName: String = "cuda:0"
) extends AbstractDeepQLearning[Id, MlpNeuralNetworkRL]
    with NodeLearner {
  import session._
  override def policy: AgentState => Int = state =>
    if (random.nextDouble() < epsilon) {
      (random.shuffle(policyNetwork.actionSpace.indices.toList).head)
    } else behaviouralPolicy(state)

  override def policyBatch: Seq[AgentState] => Seq[Int] = state => {
    val actions = policyNetwork.policyBatch(device)(state)
    actions.map { case (action) =>
      if (random.nextDouble() < epsilon) {
        (random.shuffle(policyNetwork.actionSpace.indices.toList).head)
      } else action
    }
  }
  override def store(where: String): Unit = {}

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: DecayableSource): Unit = agent.attachDecayable("epsilon" -> epsilon)

  override def encodeStates(states: Seq[Id[AgentState]]): py.Dynamic =
    referenceNet.encoder
      .encodeBatch(states.map(referenceNet.encoder.encode(_, device)), device)(identity[py.Any])
      .record()

  override def encodeActions(actions: Seq[Id[Int]]): py.Dynamic =
    torch.tensor(actions.map(action => action).toPythonCopy, device = device).record()

  override def encodeRewards(rewards: Seq[Id[Double]]): py.Dynamic = torch.nn.functional
    .normalize(torch.tensor(rewards.toPythonCopy, device = device).record(), dim = 0)
    .record()
}
