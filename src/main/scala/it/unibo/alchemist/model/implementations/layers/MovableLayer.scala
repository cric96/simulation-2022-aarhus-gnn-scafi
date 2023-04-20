package it.unibo.alchemist.model.implementations.layers

trait MovableLayer {

  /** move the layer of dx and dy */
  def moveWith(dx: Double, dy: Double): Unit

  /** return the layer to its initial position */
  def reset(): Unit
}
