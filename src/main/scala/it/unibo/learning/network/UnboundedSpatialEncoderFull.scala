package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.NeuralNetworkRL.Spatial.{encodeSpatialUnbounded, encodeSpatialUnboundedLocal}
import it.unibo.learning.network.torch.{PythonMemoryManager, geometric, torch}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

class UnboundedSpatialEncoderFull() extends GraphNeuralNetworkEncoder {
  val True = torch.tensor(Seq(true).toPythonCopy)
  override def shape: Seq[Int] = Seq(3)

  override def encodeDouble(data: Graph[Double], device: py.Any): py.Dynamic =
    geometric.data.Data(torch.tensor(data.data.toPythonCopy), encodeNeigh(data.connections, device = device))

  override def encodeInt(data: Graph[Int], device: py.Any): py.Dynamic =
    geometric.data.Data(torch.tensor(data.data.toPythonCopy), encodeNeigh(data.connections, device = device))

  override def encode(state: Graph[AgentState], device: py.Any): py.Any = {
    val data = state.data.map(state => encodeSpatialUnboundedLocal(state, considerAction = false)).toPythonCopy
    val tensor = torch.tensor(data, device = device)
    geometric.data.Data(tensor, encodeNeigh(state.connections, device = device))
  }

  override def encodeBatch(data: Seq[py.Any]): py.Dynamic = {
    val batch = geometric.loader.DataLoader(data.toPythonCopy, batch_size = data.length)
    val list = py.Dynamic.global.list(batch).bracketAccess(0)
    list
  }

  override def normalize(tensor: py.Dynamic, dimension: Int = -1): py.Dynamic = {
    val result =
      if(dimension != -1) {
        torch.nn.functional.normalize(tensor, dimension)
      } else {
        torch.nn.functional.normalize(tensor)
      }
    tensor.del()
    result
  }

  def encodeNeigh(connections: (Seq[Int], Seq[Int]), device: py.Any): py.Dynamic =
    torch.tensor(Seq(connections._1.toPythonCopy, connections._2.toPythonCopy).toPythonCopy, device = device)
}
