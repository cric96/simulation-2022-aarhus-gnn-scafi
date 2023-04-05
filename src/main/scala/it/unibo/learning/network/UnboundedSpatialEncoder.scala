package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch.{PythonMemoryManager, torch}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

class UnboundedSpatialEncoder() extends NeuralNetworkEncoder {
  val True = torch.tensor(Seq(true).toPythonCopy)
  override def shape: Seq[Int] = Seq(3)

  override def encode(state: AgentState): py.Any = {
    val x = Spatial.encodeSpatialUnbounded(state, considerAction = false)
    val index = computeNeighborhoodIndex(py.Dynamic.global.len(x).as[Int] - 1)
    (x, index)
  }

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }

  override def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit
      session: PythonMemoryManager.Session
  ): py.Dynamic = {
    import session._
    val dataRaw = scala.collection.mutable.Buffer.empty[py.Dynamic]
    val indexRawStart = scala.collection.mutable.Buffer.empty[py.Any]
    val indexRawEnd = scala.collection.mutable.Buffer.empty[py.Any]
    val maskRaw = scala.collection.mutable.Buffer.empty[py.Any]
    var max = 0
    val unsafe = seq.map(_.as[py.Dynamic])
    for (elem <- unsafe) {
      val x = elem.bracketAccess(0).as[Seq[py.Dynamic]]
      val y = elem.bracketAccess(1).as[Seq[py.Dynamic]].map(_.as[Seq[Int]])
      val yUpdated = y.map(links => links.map(_ + max))
      max = x.size + max
      dataRaw.addAll(x)
      indexRawStart.addAll(yUpdated.head.map(_.as[py.Any]))
      indexRawEnd.addAll(yUpdated.head.map(_.as[py.Any]))
      maskRaw.addAll(true :: List.fill(x.size - 1)(false).map(_.as[py.Any]))
    }
    val size = seq.head.as[py.Dynamic].bracketAccess(0).bracketAccess(0).as[Seq[Double]].size
    val x = torch.tensor(dataRaw.toPythonCopy, device = device).record().reshape(-1, size).record()
    val index =
      torch.tensor(Seq(indexRawStart.toPythonCopy, indexRawEnd.toPythonCopy).toPythonCopy, device = device).record()
    val mask = torch.tensor(maskRaw.toPythonCopy, device = device).record()
    Seq(normalize(x), index, mask).toPythonCopy.as[py.Dynamic]
  }

  def computeNeighborhoodIndex(num: Int): py.Dynamic = {
    val neighborhoodIndex =
      List(
        List(0) ::: List.fill(num)(0) ::: List.range(1, num + 1),
        List(0) ::: (List.range(1, num + 1)) ::: List.fill(num)(0)
      )
    val neighborhoodIndexPython = neighborhoodIndex.map(_.toPythonCopy).toPythonCopy
    neighborhoodIndexPython.as[py.Dynamic]
  }
}
