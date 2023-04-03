package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.collection.mutable
import scala.util.Random

trait ReplayBuffer {
  val random: Random
  def put(stateT: AgentState, actionT: Int, rewardTplus: Double, stateTPlus: AgentState): Unit
  def sample(batchSize: Int): Seq[Experience]
}
class QueueReplayBuffer(size: Int, val random: Random) {
  private val queue = mutable.Buffer.empty[Experience]
  def put(stateT: AgentState, actionT: Int, rewardTplus: Double, stateTPlus: AgentState): Unit = {
    if (queue.size == size) {
      queue.remove(0)
    }
    queue.addOne(Experience(stateT, actionT, rewardTplus, stateTPlus))
  }
  def sample(batchSize: Int): Seq[Experience] = {
    if (batchSize > queue.size)
      Seq.empty
    else
      random.shuffle(queue).take(batchSize).toSeq
  }

  def totalSize: Int = queue.size
}

class ArrayReplayBuffer(size: Int, val random: Random) {
  private val array = Array.ofDim[Experience](size)
  private var index = 0
  private var filled = false
  def put(stateT: AgentState, actionT: Int, rewardTplus: Double, stateTPlus: AgentState): Unit = {
    array(index) = Experience(stateT, actionT, rewardTplus, stateTPlus)
    if (index == size - 1)
      filled = true
    index = (index + 1) % size
  }

  def sample(batchSize: Int): Seq[Experience] = {
    if (batchSize > index && !filled) {
      Seq.empty
    } else {
      val size = if (filled) this.size else index
      var extracted = Set[Experience]()
      while (extracted.size < batchSize)
        extracted += array(random.nextInt(size))
      extracted.toSeq
    }
  }

}
object ReplayBuffer {
  case class Experience(stateT: AgentState, actionT: Int, rewardTPlus: Double, stateTPlus: AgentState)
}
