package it.unibo.alchemist.loader.`export`.extractors

import it.unibo.alchemist.loader.`export`.Extractor
import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Node, Position}
import it.unibo.scafi.space.Point2D

import java.awt.{Color, Graphics2D}
import java.awt.image.BufferedImage
import java.io.File
import java.util
import javax.imageio.ImageIO
import javax.swing.JFrame
import scala.jdk.CollectionConverters.IteratorHasAsScala

class DensityExtractor extends Extractor[Double] {
  private val size = 40 // todo move out!
  private val bound = 1000 // todo move out!
  private val precision = 10
  private val range = 100
  private val neighbor = 4
  override def getColumnNames: util.List[String] = util.List.of("density")
  val frame = new JFrame()
  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: interfaces.Time,
      l: Long
  ): util.Map[String, Double] = {
    type Aux = P forSome { type P <: Position[P] }
    val typedEnvironment = environment.asInstanceOf[Environment[T, Aux]]
    val density = environment.getLayer(new SimpleMolecule("density")).get().asInstanceOf[DensityMap[Aux]]
    var densityMap = Map.empty[(Int, Int), Double]
    var nodeComputedMap = Map.empty[(Int, Int), Double]
    for (row <- 0 until bound by precision)
      for (column <- 0 until bound by precision) {
        densityMap += (row, column) -> density.getValue(environment.typedPosition(row, column))
        val nodes = typedEnvironment
          .getNodesWithinRange(typedEnvironment.makePosition(row, column), range)
          .iterator()
          .asScala
          .toList
        val currentPosition = typedEnvironment.makePosition(row, column)
        val currentPosition2D = Point2D(currentPosition.getCoordinate(0), currentPosition.getCoordinate(1))
        val nearest = nodes
          .sortBy { node =>
            val nodePosition = typedEnvironment.getPosition(node)
            val nodePosition2D = Point2D(nodePosition.getCoordinate(0), nodePosition.getCoordinate(1))
            nodePosition2D.distance(currentPosition2D)
          }
          .take(neighbor)
        val nodeDensity =
          nearest.map(node => typedEnvironment.getPosition(node)).map(density.getValue)
        val max = nodeDensity.maxOption.getOrElse(0.0)
        val min = nodeDensity.minOption.getOrElse(0.0)
        nodeComputedMap += (row, column) -> (max + min) / 2
      }
    /*
    val image = new BufferedImage(bound, bound, BufferedImage.TYPE_INT_RGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    densityMap.foreach { case ((x, y), value) =>
      graphics.setColor(new Color(value.toInt, value.toInt, value.toInt))
      graphics.fillRect(x, y, precision, precision)
    }
    ImageIO.write(image, "png", new File("here.png"))
    val nodesImage = new BufferedImage(bound, bound, BufferedImage.TYPE_INT_RGB)
    val graphicsNodes = nodesImage.getGraphics.asInstanceOf[Graphics2D]
    nodeComputedMap.foreach { case ((x, y), value) =>
      graphicsNodes.setColor(new Color(value.toInt, value.toInt, value.toInt))
      graphicsNodes.fillRect(x, y, precision, precision)
    }

    ImageIO.write(image, "png", new File("here.png"))
    ImageIO.write(nodesImage, "png", new File("here2.png"))*/
    val error = math.sqrt(densityMap.map { case ((x, y), value) =>
      Math.pow(value - nodeComputedMap(x, y), 2)
    }.sum / densityMap.size)
    util.Map.of("density", error)
  }

  implicit class RichEnvironment[T](environment: Environment[T, _]) {
    def typedPosition(node: Node[T]): Position[_] =
      environment.getPosition(node).asInstanceOf[Position[_]]

    def typedPosition[P <: Position[P]](coordinates: Number*): P =
      environment.makePosition(coordinates: _*).asInstanceOf[P]
  }
}
