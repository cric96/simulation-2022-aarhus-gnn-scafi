package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.interfaces._
import upickle.default._

import scala.io.Source

class DensityMapFromFiles[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    val resourceFolder: String,
    val baseName: String,
    val howMany: Int,
    val changeEach: Int,
    layerMolecule: Molecule
) extends AbstractGlobalReaction[T, P](environment, timeDistribution) {
  // Internal representation to load the points
  private val allFiles = (0 to howMany).map(id => s"$resourceFolder/$baseName-$id").map(Source.fromResource(_)).toArray
  private var ticks = 0
  private var episodes = 0
  private val contents = allFiles.map(content => read[Seq[Array[Array[Double]]]](content.getLines().mkString).toArray)
  private lazy val layer = environment.getLayer(layerMolecule).get.asInstanceOf[DensityMap[P]]

  override protected def executeBeforeUpdateDistribution(): Unit = {
    layer.updateDensityMap(contents(episodes)(ticks))
    ticks += 1
    if (ticks == changeEach) {
      ticks = 0
      // episodes = (episodes + 1) % howMany
    }
  }

}
