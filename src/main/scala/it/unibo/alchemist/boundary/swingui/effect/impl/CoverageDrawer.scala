package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position2D}
import org.danilopianini.lang.RangedInteger
import org.danilopianini.view.ExportForGUI

import java.awt.geom.{AffineTransform, Ellipse2D, Path2D}
import java.awt.{BasicStroke, Color, Graphics2D}

class CoverageDrawer extends Effect {
  override def getColorSummary: Color = Color.BLACK
  override def apply[T, P <: Position2D[P]](
      g: Graphics2D,
      node: Node[T],
      environment: Environment[T, P],
      wormhole: Wormhole2D[P]
  ): Unit = {
    val manager = new SimpleNodeManager[T](node)
    val areaSize = manager.getOption[Double]("view").getOrElse(10.0).toFloat
    val center = areaSize / 2.0f
    val zoom = wormhole.getZoom
    val position = wormhole.getViewPoint(environment.getPosition(node))
    val (x, y) = (position.x, position.y)
    // Todo remove magic number pls
    g.setColor(new Color(125, 125, 125, 125))
    val shape = new Ellipse2D.Float(-center, -center, areaSize, areaSize)
    val transform = new AffineTransform
    transform.translate(x, y)
    transform.scale(zoom, zoom)
    g.fill(transform.createTransformedShape(shape))
  }
}
