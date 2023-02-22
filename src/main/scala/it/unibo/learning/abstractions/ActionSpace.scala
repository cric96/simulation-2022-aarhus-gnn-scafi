package it.unibo.learning.abstractions

object ActionSpace {
  private val minAngle = 0
  private val maxAngle = 359
  type Space = List[(Double, Double)]
  def create(definition: Int, intensities: List[Double]): Space = {
    (minAngle to maxAngle by definition).toList.flatMap { angle =>
      intensities.map(intensity => (angle, intensity))
    }
  }
}
