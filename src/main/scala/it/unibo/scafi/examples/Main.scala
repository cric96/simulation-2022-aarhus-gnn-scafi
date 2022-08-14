package it.unibo.scafi.examples

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class Main extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with BlockG with Gradients with FieldUtils {
  override def main(): Double = senseEnvData("density")
}