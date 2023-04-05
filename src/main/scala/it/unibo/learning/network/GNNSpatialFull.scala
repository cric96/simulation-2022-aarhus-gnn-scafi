package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

class GNNSpatialFull(
    hiddenSize: Int,
    val actionSpace: List[Any],
    val encoder: GraphNeuralNetworkEncoder = new UnboundedSpatialEncoderFull()
) extends GraphNeuralNetworkRL {

  override val underlying: py.Dynamic = GNNDQN(encoder.shape.head, hiddenSize, actionSpace.size)

  override def cloneNetwork: GraphNeuralNetworkRL = new GNNSpatialFull(hiddenSize, actionSpace, encoder)

  override def policy(device: py.Any): Graph[AgentState] => Graph[Int] = {
    implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()
    import session._
    val _underlying = underlying
    val _device = device
    (graph: Graph[AgentState]) => {
      py.`with`(torch.no_grad()) { _ =>
        val encoded = encoder.encode(graph, device).asInstanceOf[py.Dynamic]
        val normalized = encoder.normalize(encoded.x).record()
        val tensor = normalized.to(_device)
        val output = _underlying.forward(tensor, encoded.edge_index.to(_device)).record()
        val action = output
          .max(1)
          .record()
          .bracketAccess(1)
          .record()
        val actionList = action.tolist().as[Seq[Int]]
        Graph(actionList, graph.connections)
      }
    }
  }
}
