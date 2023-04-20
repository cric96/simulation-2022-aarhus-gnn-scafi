package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Layer, Position, Position2D}

trait CenterBasedMovableLayer[P <: Position2D[P]] extends MovableLayer with Layer[Double, P] {
  def x: Double // x component of the center
  def y: Double // y component of the center

  def value: Double // value of the layer

  protected var movingX: Double = x
  protected var movingY: Double = y

  override def getValue(p: P): Double = {
    // check if the point is inside the circle
    val pointX = p.getX
    val pointY = p.getY
    if (isInside(pointX, pointY)) {
      value
    } else {
      0
    }
  }

  protected def isInside(x: Double, y: Double): Boolean

  override def moveWith(dx: Double, dy: Double): Unit = {
    movingX += dx
    movingY += dy
  }

  override def reset(): Unit = {
    movingX = x
    movingY = y
  }
}
