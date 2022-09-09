package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Layer, Position}
import smile.neighbor.{KDTree, Neighbor}

import java.util

class DensityMap[P <: Position[P]](range: Double) extends Layer[Double, P] {
  private var index: Option[KDTree[Object]] = None

  override def getValue(p: P): Double = {
    index
      .map { index =>
        val result = new util.ArrayList[Neighbor[Array[Double], Object]]()
        index.range(p.getCoordinates, range, result)
        result.size()
      }
      .getOrElse[Int](0)
  }
  def updateDensityMap(points: Array[Array[Double]]): Unit = {
    val elementsEmpty: Array[Object] = points.map(a => a: Object)
    index = Some(new KDTree(points, elementsEmpty))
  }
}
