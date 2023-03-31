package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class MLPSpatial(hiddenSize: Int, val actionSpace: List[Any], val encoder: NeuralNetworkEncoder = new SpatialEncoder(5))
    extends NeuralNetworkRL {
  override val underlying: py.Dynamic = DQN(this.encoder.shape.reverse.head, hiddenSize, actionSpace.size)

  override def policy(device: py.Any): (AgentState) => Int =
    NeuralNetworkRL.policyFromNetwork(this, this.encoder.shape, device)

  override def cloneNetwork: NeuralNetworkRL = new MLPSpatial(hiddenSize, actionSpace, encoder)

  override def policyBatch(device: py.Any): Seq[AgentState] => Seq[Int] =
    NeuralNetworkRL.policyFromNetworkBatch(this, encoder.shape, device)

}
