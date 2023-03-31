package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Layer, Position2D}

class SquareLayer[P <: Position2D[P]](
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double,
    val value: Double
) extends Layer[Double, P] {
  override def getValue(p: P): Double = {
    // check if the point is inside the square
    val pointX = p.getX
    val pointY = p.getY
    if (pointX >= x && pointX <= x + width && pointY >= y && pointY <= y + height) {
      value
    } else {
      0
    }
  }
}
