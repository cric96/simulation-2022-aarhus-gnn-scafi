package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Layer, Position2D}

// Similar to src/main/scala/it/unibo/alchemist/model/implementations/layers/SquareLayer.scala but for a circle
class CircleLayer[P <: Position2D[P]](
    val x: Double,
    val y: Double,
    val radius: Double,
    val value: Double
) extends Layer[Double, P] {
  override def getValue(p: P): Double = {
    // check if the point is inside the circle
    val pointX = p.getX
    val pointY = p.getY
    if (Math.sqrt(Math.pow(pointX - x, 2) + Math.pow(pointY - y, 2)) <= radius) {
      value
    } else {
      0
    }
  }
}
