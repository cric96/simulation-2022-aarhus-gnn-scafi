package it.unibo.learning

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.interfaces.{Environment, Position}
import org.apache.commons.math3.random.RandomGenerator

case class Box(width: Double, step: Double, randomness: Double)

object Box {
  implicit class BoxOps(box: Box) {
    def createGrid[T, P <: Position[P]](environment: Environment[T, P], random: RandomGenerator): Grid = new Grid(
      environment,
      random,
      0.0f,
      0.0f,
      box.width,
      box.width,
      box.step,
      box.step,
      box.randomness
    )
  }
}
