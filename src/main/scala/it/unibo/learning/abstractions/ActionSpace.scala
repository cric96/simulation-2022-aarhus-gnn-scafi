package it.unibo.learning.abstractions

object ActionSpace {
  private val minAngle = 0
  private val maxAngle = 359
  type Space = List[(Double, Double)]
  def create(definition: Int, intensities: List[Double], standStill: Boolean = false): Space = {
    require(intensities.forall(_ >= 0))
    val standStillVelocity: List[(Double, Double)] = if (standStill) { List((0.0, 0.0)) }
    else List.empty
    (minAngle to maxAngle by definition).toList.flatMap { angle =>
      intensities.map(intensity => (angle.toDouble, intensity))
    } ::: standStillVelocity
  }
}
