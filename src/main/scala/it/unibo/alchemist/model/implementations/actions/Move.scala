package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.abstractions.Contextual

class Move[P <: Position[P]](node: Node[Any], environment: Environment[Any, P], actionSpace: List[(Double, Double)])
    extends AbstractLocalAction[Any](node) {
  private val manager = new SimpleNodeManager(node)
  override def cloneAction(node: Node[Any], reaction: Reaction[Any]): Action[Any] =
    new Move(node, environment, actionSpace)

  override def execute(): Unit = {
    manager.getOption[(Int, Contextual)]("action") match {
      case Some((index, _)) =>
        val (angle, module) = actionSpace(index)
        manager.put("angle", angle)
        manager.put("intensity", module)
        val deltaVector = (module * math.cos(angle), module * math.sin(angle))
        environment.moveNodeToPosition(
          node,
          environment.getPosition(node).plus(Array(deltaVector._1 * 10, -deltaVector._2 * 10))
        )
    }
  }
}
