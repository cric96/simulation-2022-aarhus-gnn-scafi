package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.{PythonMemoryManager, geometric, torch}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}

trait GraphNeuralNetworkEncoder {
  def shape: Seq[Int] // node feature set
  def encodeDouble(data: Graph[Double], device: py.Any): py.Dynamic
  def encodeInt(data: Graph[Int], device: py.Any): py.Dynamic
  def encode(state: Graph[AgentState], device: py.Any): py.Any
  def encodeBatch[A: Reader](data: Seq[py.Any], device: py.Any)(conversion: A => py.Any): py.Dynamic

  def encodeBatchNormalize[A: Reader](data: Seq[py.Any], device: py.Any, dimension: Int = -1)(conversion: A => py.Any): py.Dynamic = {
    manipulateGraphInLocal {
      val batch = encodeBatch(data, device)(conversion)
      val X = normalize(batch.x, dimension)
      (X.tolist().as[Seq[A]], batch.edge_index.tolist().as[Seq[Seq[Int]]])
    }(device, conversion)
  }

  def manipulateGraphInLocal[A: Reader](block: => (Seq[A], Seq[Seq[Int]]))(device: py.Any, conversion: A => py.Any): py.Dynamic = {
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

  def normalize(tensor: py.Dynamic, dimension: Int = -1): py.Dynamic
}
