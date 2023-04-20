package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.MlpNeuralNetworkRL.Spatial
import it.unibo.learning.network.torch._
import it.unibo.scripting.Unsafe
import it.unibo.typelevel.Id
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class MLPSpatial(
    hiddenSize: Int,
    val actionSpace: List[Any],
    val encoder: NeuralNetworkEncoder[Id] = new SpatialEncoder(5)
) extends MlpNeuralNetworkRL {
  override val underlying: py.Dynamic = DQN(this.encoder.shape.reverse.head, hiddenSize, actionSpace.size)

  override def policy(device: py.Any): (AgentState) => Int = state => {
    val netInput = encoder.encode(state, device)
    val session = PythonMemoryManager.session()
    // context
    import session._
    py.`with`(torch.no_grad()) { _ =>
      val tensor = torch
        .tensor(netInput)
        .record()
        .applyDynamic("view")(this.encoder.shape.map(_.as[py.Any]): _*)
        .record()
        .to(device)
      val normalized = nn.encoder.normalize(tensor).record()
      val netOutput = nn.underlying(normalized).record()
      val elements = netOutput.tolist().record().bracketAccess(0).record()
      val max = py.Dynamic.global.max(elements)
      val index = elements.index(max).as[Int]
      session.clear()
      index
    }
  }

  override def cloneNetwork: MlpNeuralNetworkRL = new MLPSpatial(hiddenSize, actionSpace, encoder)

  override def policyBatch(device: py.Any): Seq[AgentState] => Seq[Int] =
    MlpNeuralNetworkRL.policyFromNetworkBatch(this, encoder.shape, device)

}
