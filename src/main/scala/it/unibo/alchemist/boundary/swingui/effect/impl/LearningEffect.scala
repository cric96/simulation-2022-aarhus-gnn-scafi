package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.swingui.effect.impl.LearningEffect._
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.implementations.nodes._
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position2D}
import it.unibo.scafi.space.Point3D
import org.danilopianini.lang.RangedInteger
import org.danilopianini.view.ExportForGUI

import java.awt._
import java.awt.geom._

/** ad-hoc effect to draw drones, animal and station. */
class LearningEffect extends Effect {
  private var positionsMemory: Map[Int, Seq[(Position2D[_], Double)]] = Map.empty
  private var lastDrawMemory: Map[Int, Int] = Map.empty

  @ExportForGUI(nameToExport = "Track")
  private val trackEnabled: Boolean = true

  @ExportForGUI(nameToExport = "SnapshotSize")
  val snapshotSize: RangedInteger = new RangedInteger(10, MAX_LENGTH, LENGTH)

  @ExportForGUI(nameToExport = "SnapshotFrequency")
  val timespan: RangedInteger = new RangedInteger(1, 100, CLOCK)

  @ExportForGUI(nameToExport = "NodeSize")
  val nodeSize: RangedInteger = new RangedInteger(1, 20, DRONE_SIZE.toInt)

  override def apply[T, P <: Position2D[P]](
      g: Graphics2D,
      node: Node[T],
      env: Environment[T, P],
      wormhole: Wormhole2D[P]
  ): Unit = {
    val nodePosition: P = env.getPosition(node)
    val viewPoint: Point = wormhole.getViewPoint(nodePosition)
    val (x, y) = (viewPoint.x, viewPoint.y)
    drawArrow(g, node, x, y, env, wormhole)
  }

  def getColorSummary: Color = Color.BLACK

  def drawArrow[T, P <: Position2D[P]](
      g: Graphics2D,
      droneNode: Node[T],
      x: Int,
      y: Int,
      env: Environment[T, P],
      wormhole: Wormhole2D[P]
  ): Unit = {
    val currentRotation = rotation(droneNode)
    val transform = getTransform(x, y, nodeSize.getVal, currentRotation)
    val shape = DRONE_SHAPE
    val color = droneColor(droneNode.getId, env.getNodeCount)
    val transformedShape = transform.createTransformedShape(shape)
    val positions = positionsMemory.getOrElse(droneNode.getId, Seq.empty)
    val lastDraw = lastDrawMemory.getOrElse(droneNode.getId, 0)
    val alpha = MAX_COLOR / ((Math.min(snapshotSize.getVal, positions.size) * ADJUST_ALPHA_FACTOR) + 1)

    positions.filter(_ => trackEnabled).takeRight(snapshotSize.getVal).zipWithIndex.foreach {
      case ((nodePosition, rotation), index) =>
        val colorFaded =
          new Color(color.getRed, color.getGreen, color.getBlue, Math.max(1, (alpha * (index + 1)).toInt))
        val viewPoint: Point = wormhole.getViewPoint(nodePosition.asInstanceOf[P])
        val (x, y) = (viewPoint.x, viewPoint.y)
        val transform = getTransform(x, y, nodeSize.getVal, rotation)
        val transformedShape = transform.createTransformedShape(shape)
        g.setColor(colorFaded)
        g.fill(transformedShape)
    }
    g.setColor(color)
    g.fill(transformedShape)

    val roundedTick = env.getSimulation.getTime.toDouble.toInt
    if (roundedTick >= lastDraw) {
      lastDrawMemory += (droneNode.getId -> (lastDraw + timespan.getVal))
      positionsMemory += (droneNode.getId -> (positions :+ (env.getPosition(droneNode), currentRotation))
        .takeRight(MAX_LENGTH))
    }
  }

  private def rotation[T](node: Node[T]): Double = {
    val nodeManager = new SimpleNodeManager[T](node)
    val velocity = nodeManager.getOption("velocity").getOrElse(Point3D(1, 0, 0))
    math.atan2(velocity.x, velocity.y)
  }

  private def getTransform(x: Int, y: Int, zoom: Double, rotation: Double): AffineTransform = {
    val transform = new AffineTransform()
    transform.translate(x, y)
    transform.scale(zoom, zoom)
    transform.rotate(rotation)
    transform
  }
}

object LearningEffect {
  private val ADJUST_ALPHA_FACTOR: Int = 4

  private val CLOCK: Int = 10

  private val LENGTH: Int = 140

  private val MAX_LENGTH: Int = 1000

  private val MAX_COLOR: Double = 255

  private val DRONE_SHAPE: Polygon = new Polygon(Array(-1, 0, 1), Array(2, -2, 2), 3)

  private val DRONE_SIZE = 4.0
  private def droneColor(id: Int, howMany: Int): Color = new Color(
    Color.HSBtoRGB(id.toFloat / howMany.toFloat, 1, 0.8f)
  )
}
