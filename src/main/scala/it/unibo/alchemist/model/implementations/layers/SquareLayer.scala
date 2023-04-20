package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Layer, Position2D}

class SquareLayer[P <: Position2D[P]](
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double,
    val value: Double
) extends Layer[Double, P] with MovableLayer {
  var movingX = x
  var movingY = y
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

  /** move the layer of dx and dy */
  override def moveWith(dx: Double, dy: Double): Unit = {
    movingX += dx
    movingY += dy
  }

  /** return the layer to its initial position */
  override def reset(): Unit = {
    movingX = x
    movingY = y
  }
}
