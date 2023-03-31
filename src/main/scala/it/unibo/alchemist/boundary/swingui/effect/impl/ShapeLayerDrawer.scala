package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.DrawLayers
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.interfaces.{Environment, Layer, Position2D}

import java.awt.{Color, Graphics2D}
import java.util

class ShapeLayerDrawer extends DrawLayers {
  override def drawLayers[T, P <: Position2D[P]](
      toDraw: util.Collection[Layer[T, P]],
      environment: Environment[T, P],
      graphics: Graphics2D,
      wormhole: Wormhole2D[P]
  ): Unit = ???

  override def getColorSummary: Color = Color.blue
}
