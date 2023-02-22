package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.collection.mutable
import scala.util.Random

class ReplayBuffer(size: Int, random: Random) {
  private val queue = mutable.Buffer.empty[Experience]
  def put(stateT: AgentState, actionT: Int, rewardTplus: Double, stateTPlus: AgentState): Unit = {
    if (queue.size == size) {
      queue.remove(0)
    }
    queue.addOne(Experience(stateT, actionT, rewardTplus, stateTPlus))
  }
  def sample(batchSize: Int): Seq[Experience] =
    random.shuffle(queue).take(batchSize).toSeq

  def totaleSize: Int = queue.size
}
object ReplayBuffer {
  case class Experience(stateT: AgentState, actionT: Int, rewardTPlus: Double, stateTPlus: AgentState)
}
