package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch._
import it.unibo.learning.network.torch.{util => torchUtil}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import scribe._

import scala.collection.mutable
class GNNSpatialFull(
    hiddenSize: Int,
    val actionSpace: List[Any],
    val encoder: NeuralNetworkEncoder[Graph] = new UnboundedSpatialEncoderFull()
) extends GraphNeuralNetworkRL {

  override val underlying: py.Dynamic = GNNDQN(encoder.shape.head, hiddenSize, actionSpace.size)

  override def cloneNetwork: GraphNeuralNetworkRL = new GNNSpatialFull(hiddenSize, actionSpace, encoder)

  override def policy(device: py.Any): Graph[AgentState] => Graph[Int] = {
    implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()
    import session._
    val _device = device
    (graph: Graph[AgentState]) => {
      py.`with`(torch.no_grad()) { _ =>
        scribe.info(s"Before forward pass : ${torchUtil.dumbMemory(device)}")
        val encoded = encoder.encode(graph, _device).asInstanceOf[py.Dynamic]
        encoded.record()
        val normalized = encoder.normalize(encoded.x).record()
        val tensor = normalized.to(_device).record()
        val output = underlying(tensor, encoded.edge_index.record().to(_device).record()).record()
        val buffer = mutable.Buffer.empty[Int]
        py.local {
          val data = py"${output}.max(1)[1].tolist()"
          for(i <- 0 until py.Dynamic.global.len(data).as[Int]) {
            buffer += data.bracketAccess(i).as[Int]
          }
        }
        scribe.info(s"After forward pass : ${torchUtil.dumbMemory(device)}")
        session.clear()
        scribe.info(s"Clear after forward pass : ${torchUtil.dumbMemory(device)}")
        Graph(buffer.toSeq, graph.connections)
      }
    }

  }
}
