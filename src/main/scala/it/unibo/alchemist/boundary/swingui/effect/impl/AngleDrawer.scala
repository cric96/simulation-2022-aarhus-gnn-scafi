package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position2D}
import org.danilopianini.lang.RangedInteger
import org.danilopianini.view.ExportForGUI

import java.awt.geom.{AffineTransform, Path2D}
import java.awt.{BasicStroke, Color, Graphics2D}

class AngleDrawer extends Effect {
  @ExportForGUI(nameToExport = "Melecule")
  private val molecule: String = "angle"
  @ExportForGUI(nameToExport = "Size")
  private val maxValueRanged: RangedInteger = new RangedInteger(10, 100, 20)
  private def maxValue = maxValueRanged.getVal
  @ExportForGUI(nameToExport = "Stroke")
  private val stroke: RangedInteger = new RangedInteger(1, 10, 2)
  private def strokeValue = stroke.getVal
  override def getColorSummary: Color = Color.BLACK

  override def apply[T, P <: Position2D[P]](g: Graphics2D, node: Node[T], environment: Environment[T, P], wormhole: Wormhole2D[P]): Unit = {
    val manager = new SimpleNodeManager[T](node)
    val angle = manager.getOption[Double](molecule).getOrElse(0.0)
    val direction = (math.cos(angle), math.sin(angle))
    val zoom = wormhole.getZoom
    val position = wormhole.getViewPoint(environment.getPosition(node))
    val (x, y) = (position.x, position.y)
    g.setColor(Color.BLACK)
    g.setStroke(new BasicStroke(strokeValue))
    val path = new Path2D.Float()
    path.moveTo(0, 0)
    path.lineTo(math.cos(angle) , math.sin(angle))
    val transform = new AffineTransform
    transform.translate(x, y)
    transform.scale(maxValue, maxValue)
    transform.scale(zoom, zoom)
    g.draw(transform.createTransformedShape(path))
  }
}