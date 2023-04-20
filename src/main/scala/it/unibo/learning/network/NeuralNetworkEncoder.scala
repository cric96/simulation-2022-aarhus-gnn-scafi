package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.PythonMemoryManager
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.Reader

/*trait NeuralNetworkEncoder {
  def shape: Seq[Int]
  def encode(state: AgentState): py.Any
  def normalize(tensor: py.Dynamic): py.Dynamic
  def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit session: PythonMemoryManager.Session): py.Dynamic
}
 */

trait NeuralNetworkEncoder[F[_]] {
  def unsafe: Boolean

  def shape: Seq[Int] // node feature set

  def encodeDouble(data: F[Double], device: py.Any): py.Dynamic

  def encodeInt(data: F[Int], device: py.Any): py.Dynamic

  def encode(state: F[AgentState], device: py.Any): py.Any

  def encodeBatch[A: Reader](data: Seq[py.Any], device: py.Any)(conversion: A => py.Any): py.Dynamic

  def encodeBatchNormalize[A: Reader](data: Seq[py.Any], device: py.Any, dimension: Int = -1)(
      conversion: A => py.Any
  ): py.Dynamic = normalize(encodeBatch(data, dimension)(conversion), dimension)

  def normalize(tensor: py.Dynamic, dimension: Int = -1): py.Dynamic
}
