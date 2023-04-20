package it.unibo.alchemist.model.implementations.layers

import it.unibo.alchemist.model.interfaces.{Environment, Layer, Position}

import scala.jdk.CollectionConverters.IteratorHasAsScala

class CombineAllLayer[P <: Position[P]](environment: Environment[Double, P]) extends Layer[Double, P] {
  override def getValue(p: P): Double = environment.getLayers.iterator().asScala.filter(_ != this).map(_.getValue(p)).sum
}
