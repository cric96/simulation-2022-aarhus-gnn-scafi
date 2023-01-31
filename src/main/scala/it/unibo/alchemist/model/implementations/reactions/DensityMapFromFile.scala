package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.interfaces._
import org.danilopianini.util.{ListSet, ListSets}

import java.util
import scala.io.Source
import upickle.default._

class DensityMapFromFile[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    val resource: String,
    layerMolecule: Molecule
) extends AbstractGlobalReaction[T, P](environment, timeDistribution) {
  // Internal representation to load the points
  private val resourceContent = Source.fromResource(resource)
  private var content = read[Seq[Array[Array[Double]]]](resourceContent.getLines().mkString)

  private lazy val layer = environment.getLayer(layerMolecule).get.asInstanceOf[DensityMap[P]]

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val current = content.head
    content = content.tail
    layer.updateDensityMap(current)
  }
}
