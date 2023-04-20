package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.DecayableSource
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, DecayReference}
import it.unibo.learning.network.torch._
import it.unibo.learning.network.{Graph, GraphNeuralNetworkRL}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.util.Random

class DeepGnnQLearning(
    override val epsilon: DecayReference[Double],
    override val alpha: Double,
    override val gamma: Double,
    override val copyEach: Int,
    override val referenceNet: GraphNeuralNetworkRL,
    override val deviceName: String = "cpu"
) extends AbstractDeepQLearning[Graph, GraphNeuralNetworkRL]
    with GraphLearner {
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

  import session._

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: DecayableSource): Unit = agent.attachDecayable("epsilon" -> epsilon)

  override def encodeStates(states: Seq[Graph[AgentState]]): py.Dynamic = encoder
    .encodeBatchNormalize[Seq[Double]](states.map(encoder.encode(_, device = device).record()), device)(
      _.toPythonCopy
    )
    .record()

  override def encodeActions(actions: Seq[Graph[Int]]): py.Dynamic = referenceNet.encoder
    .encodeBatch[Int](actions.map(referenceNet.encoder.encodeInt(_, device).record()), device)(
      identity[Int]
    )
    .record()
    .x
    .record()

  override def encodeRewards(rewards: Seq[Graph[Double]]): py.Dynamic = {
    val raw = referenceNet.encoder
      .encodeBatch[Double](rewards.map(referenceNet.encoder.encodeDouble(_, device).record()), device)(
        identity[Double]
      )
      .record()
      .x
      .record()
    torch.nn.functional.normalize(raw, dim = 0).record().to(device).record()
  }
}
