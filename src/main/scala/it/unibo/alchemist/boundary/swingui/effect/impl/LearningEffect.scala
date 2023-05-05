package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.swingui.effect.impl.LearningEffect._
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.nodes._
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position2D}
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
  private val snapshotSize: RangedInteger = new RangedInteger(10, MAX_LENGTH, LENGTH)

  @ExportForGUI(nameToExport = "SnapshotFrequency")
  private val timespan: RangedInteger = new RangedInteger(1, 100, CLOCK)

  @ExportForGUI(nameToExport = "NodeSize")
  private val nodeSize: RangedInteger = new RangedInteger(1, 20, DRONE_SIZE.toInt)

  @ExportForGUI(nameToExport = "Hue Molecule Property")
  private var colorMolecule: String = "hue"

  @ExportForGUI(nameToExport = "Velocity Molecule Property")
  private var velocityMolecule: String = "velocity"

  @ExportForGUI(nameToExport = "Max Value")
  private var maxValue: String = ""

  override def apply[T, P <: Position2D[P]](
      g: Graphics2D,
      node: Node[T],
      env: Environment[T, P],
      wormhole: Wormhole2D[P]
  ): Unit = {
    val nodePosition: P = env.getPosition(node)
    val viewPoint: Point = wormhole.getViewPoint(nodePosition)
    val (x, y) = (viewPoint.x, viewPoint.y)
    drawDirectedNode(g, node, x, y, env, wormhole)
  }

  def getColorSummary: Color = Color.BLACK

  def drawDirectedNode[T, P <: Position2D[P]](
      g: Graphics2D,
      node: Node[T],
      x: Int,
      y: Int,
      environment: Environment[T, P],
      wormhole: Wormhole2D[P]
  ): Unit = {
    val currentRotation = rotation(node)
    val transform = getTransform(x, y, nodeSize.getVal, currentRotation)
    val color = createColorOrBlack(node, environment)
    val transformedShape = transform.createTransformedShape(DRONE_SHAPE)
    if (trackEnabled) drawTrajectory(node, color, wormhole, g, DRONE_SHAPE)
    g.setColor(color)
    g.fill(transformedShape)
    updateTrajectory(node, environment)
  }

  private def rotation[T](node: Node[T]): Double = {
    val nodeManager = new SimpleNodeManager[T](node)
    val velocity = nodeManager.getOption(velocityMolecule).getOrElse(Array(1.0, 0.0))
    math.atan2(velocity(0), velocity(1))
  }

  private def getTransform(x: Int, y: Int, zoom: Double, rotation: Double): AffineTransform = {
    val transform = new AffineTransform()
    transform.translate(x, y)
    transform.scale(zoom, zoom)
    transform.rotate(rotation)
    transform
  }

  private def drawTrajectory[P <: Position2D[P]](
      node: Node[_],
      colorBase: Color,
      wormhole: Wormhole2D[P],
      g: Graphics2D,
      shape: Shape
  ): Unit = {
    val positions = positionsMemory.getOrElse(node.getId, Seq.empty)
    val alpha = MAX_COLOR / ((Math.min(snapshotSize.getVal, positions.size) * ADJUST_ALPHA_FACTOR) + 1)
    positions.filter(_ => trackEnabled).takeRight(snapshotSize.getVal).zipWithIndex.foreach {
      case ((nodePosition, rotation), index) =>
        val colorFaded =
          new Color(colorBase.getRed, colorBase.getGreen, colorBase.getBlue, Math.max(1, (alpha * (index + 1)).toInt))
        val viewPoint: Point = wormhole.getViewPoint(nodePosition.asInstanceOf[P])
        val (x, y) = (viewPoint.x, viewPoint.y)
        val transform = getTransform(x, y, nodeSize.getVal, rotation)
        val transformedShape = transform.createTransformedShape(shape)
        g.setColor(colorFaded)
        g.fill(transformedShape)
    }
  }

  private def updateTrajectory[P <: Position2D[P], T](node: Node[T], environment: Environment[T, P]): Unit = {
    val positions = positionsMemory.getOrElse(node.getId, Seq.empty)
    val lastDraw = lastDrawMemory.getOrElse(node.getId, 0)
    val roundedTime = environment.getSimulation.getTime.toDouble.toInt
    if (roundedTime >= lastDraw) {
      lastDrawMemory += (node.getId -> (lastDraw + timespan.getVal))
      positionsMemory += (node.getId ->
        (positions :+ (environment.getPosition(node) -> rotation(node)))
          .takeRight(MAX_LENGTH))
    }
  }
  private def createColorOrBlack(node: Node[_], environment: Environment[_, _]): Color = {
    val currentMolecule = new SimpleMolecule(colorMolecule)
    if (node.contains(currentMolecule)) {
      val hue = node.getConcentration(currentMolecule).asInstanceOf[Number].doubleValue()
      val hueComponent = (hue.toFloat / maxValue.toDoubleOption.getOrElse(environment.getNodeCount.toDouble)).toFloat
      new Color(Color.HSBtoRGB(hueComponent, 1, 0.8f))
    } else {
      Color.BLACK
    }
  }
}

object LearningEffect {
  private val ADJUST_ALPHA_FACTOR: Int = 4

  private val CLOCK: Int = 10

  private val LENGTH: Int = 140

  private val MAX_LENGTH: Int = 1000

  private val MAX_COLOR: Double = 255

  private val DRONE_SHAPE: Polygon = new Polygon(Array(-1, 0, 1), Array(1, -2, 1), 3)

  private val DRONE_SIZE = 4.0
}
