package it.unibo.scafi.examples

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class Clustering extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with BlockG with Gradients with FieldUtils {
  lazy val color = alchemistRandomGen.nextInt(3)
  override def main(): Double = {
    val result2 = rep((color, 1)) {
      case (color) =>
        val map = includingSelf.reifyField(nbr(color))
        val aggregation = map.groupBy(_._2._1).map { case (label, elements) =>
          (label, elements.size, elements.map(_._2._2).sum)
        }
        node.put("aggregation", aggregation)
        val winner = aggregation.maxBy(_._3)
        (winner._1, winner._2)
    }
    val result = rep(color) { color =>
      val map = includingSelf.reifyField(nbr(color))
      map.groupBy(_._2).map { case (label, elements) => (label, elements.size)}.maxBy(_._2)._1
    }
    node.put("data",  result2)
    result
  }
}