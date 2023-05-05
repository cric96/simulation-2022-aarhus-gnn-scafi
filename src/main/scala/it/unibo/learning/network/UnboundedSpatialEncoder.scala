package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.MlpNeuralNetworkRL.Spatial
import it.unibo.learning.network.MlpNeuralNetworkRL.Spatial.{encodeSpatialUnbounded, encodeSpatialUnboundedLocal}
import it.unibo.learning.network.torch.{PythonMemoryManager, geometric, torch}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import me.shadaj.scalapy.readwrite.{Reader, Writer}

class UnboundedSpatialEncoder(override val unsafe: Boolean = true) extends NeuralNetworkEncoder[Graph] {
  val True = torch.tensor(Seq(true).toPythonCopy)
  override def shape: Seq[Int] = Seq(3)

  override def encodeDouble(data: Graph[Double], device: py.Any): py.Dynamic = {
    val X = torch.tensor(data.data.toPythonCopy)
    val edge = encodeNeigh(data.connections, device = device)
    val result = geometric.data.Data(X, edge)
    X.del()
    edge.del()
    result
  }

  override def encodeInt(data: Graph[Int], device: py.Any): py.Dynamic = {
    val X = torch.tensor(data.data.toPythonCopy)
    val edge = encodeNeigh(data.connections, device = device)
    val result = geometric.data.Data(X, edge)
    X.del()
    edge.del()
    result
  }

  override def encode(state: Graph[AgentState], device: py.Any): py.Any = {
    val data = state.data.map(state => encodeSpatialUnboundedLocal(state, considerAction = false)).toPythonCopy
    val tensor = torch.tensor(data, device = device)
    val neighborhood = encodeNeigh(state.connections, device = device)
    val result = geometric.data.Data(tensor, neighborhood)
    tensor.del()
    neighborhood.del()
    result
  }

  override def encodeBatchNormalize[A: Reader](data: Seq[py.Any], device: py.Any, dimension: Int = -1)(
      conversion: A => py.Any
  ): py.Dynamic = {
    val batch = encodeBatch(data, device)(conversion)
    if (unsafe) {
      geometric.data.Data(normalize(batch.x, dimension), batch.edge_index)
    } else {
      manipulateGraphInLocal {
        val X = normalize(batch.x, dimension)
        (X.tolist().as[Seq[A]], batch.edge_index.tolist().as[Seq[Seq[Int]]])
      }(device, conversion)
    }
  }

  def manipulateGraphInLocal[A: Reader](
      block: => (Seq[A], Seq[Seq[Int]])
  )(device: py.Any, conversion: A => py.Any): py.Dynamic = {
    val (batchedX, batchedEdge) = py.local {
      block
    }
    val X = torch.tensor(batchedX.map(conversion).toPythonCopy, device = device)
    val edge = torch.tensor(batchedEdge.map(_.toPythonCopy).toPythonCopy, device = device)
    val result = geometric.data.Data(X, edge)
    X.del()
    edge.del()
    result
  }
  override def encodeBatch[A: Reader](data: Seq[py.Any], device: py.Any)(conversion: A => py.Any): py.Dynamic = {
    if (unsafe) {
      val iterable = geometric.loader.DataLoader(data.toPythonCopy, batch_size = data.length)
      py.Dynamic.global.list(iterable).bracketAccess(0)
    } else {
      manipulateGraphInLocal {
        val iterable = py.Dynamic.global.iter(geometric.loader.DataLoader(data.toPythonCopy, batch_size = data.length))
        val extracted = py"[x for x in $iterable][0]"
        (extracted.x.tolist().as[Seq[A]], extracted.edge_index.tolist().as[Seq[Seq[Int]]])
      }(device, conversion)
    }
  }

  override def normalize(tensor: py.Dynamic, dimension: Int = -1): py.Dynamic = {
    val result =
      if (dimension != -1) {
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
