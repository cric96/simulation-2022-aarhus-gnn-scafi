package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.{PythonMemoryManager, geometric}
import me.shadaj.scalapy.py

trait GraphNeuralNetworkEncoder {
  def shape: Seq[Int] // node feature set
  def encodeDouble(data: Graph[Double], device: py.Any): py.Dynamic
  def encodeInt(data: Graph[Int], device: py.Any): py.Dynamic
  def encode(state: Graph[AgentState], device: py.Any): py.Any
  def encodeBatch(data: Seq[py.Any]): py.Dynamic

  def encodeBatchNormalize(data: Seq[py.Any], dimension: Int = -1): py.Dynamic = {
    val batch = encodeBatch(data)
    val X = normalize(batch.x, dimension)
    geometric.data.Data(X, edge_index = batch.edge_index)
  }
  def normalize(tensor: py.Dynamic, dimension: Int = -1): py.Dynamic
}
