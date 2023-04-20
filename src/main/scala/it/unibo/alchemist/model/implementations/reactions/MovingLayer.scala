package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.{CenterBasedMovableLayer, CircleLayer}
import it.unibo.alchemist.model.interfaces.{Environment, Position, TimeDistribution}

import scala.jdk.CollectionConverters.IteratorHasAsScala

class MovingLayer[T, P <: Position[P]](
    val environment: Environment[T, P],
    val distribution: TimeDistribution[T],
    val period: Int
) extends AbstractGlobalReaction[T, P] {
  override protected def executeBeforeUpdateDistribution(): Unit = {
    environment.getLayers.iterator().asScala.foreach {
      case circle: CenterBasedMovableLayer[_] =>
        circle.moveWith(0.2, 0.2)
        if (environment.getSimulation.getTime.toDouble.toInt % period == 0) {
          circle.reset()
        }
      case _ =>
    }
  }
}
