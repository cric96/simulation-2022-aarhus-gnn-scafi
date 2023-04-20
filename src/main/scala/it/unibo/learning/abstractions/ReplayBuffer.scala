package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.collection.mutable
import scala.util.Random

trait ReplayBuffer[F[_]] {
  val random: Random
  def put(stateT: F[AgentState], actionT: F[Int], rewardTplus: F[Double], stateTPlus: F[AgentState]): Unit
  def sample(batchSize: Int): Seq[Experience[F]]
}
class QueueReplayBuffer[F[_]](size: Int, val random: Random) extends ReplayBuffer[F] {
  private val queue = mutable.Buffer.empty[Experience[F]]
  def put(stateT: F[AgentState], actionT: F[Int], rewardTplus: F[Double], stateTPlus: F[AgentState]): Unit = {
    if (queue.size == size) {
      queue.remove(0)
    }
    queue.addOne(Experience(stateT, actionT, rewardTplus, stateTPlus))
  }
  def sample(batchSize: Int): Seq[Experience[F]] = {
    if (batchSize > queue.size)
      Seq.empty
    else
      random.shuffle(queue).take(batchSize).toSeq
  }
}

class ArrayReplayBuffer[F[_]](size: Int, val random: Random) extends ReplayBuffer[F]{
  private val array = Array.ofDim[Experience[F]](size)
  private var index = 0
  private var filled = false
  def put(stateT: F[AgentState], actionT: F[Int], rewardTplus: F[Double], stateTPlus: F[AgentState]): Unit = {
    array(index) = Experience(stateT, actionT, rewardTplus, stateTPlus)
    if (index == size - 1)
      filled = true
    index = (index + 1) % size
  }

  def sample(batchSize: Int): Seq[Experience[F]] = {
    if (batchSize > index && !filled) {
      Seq.empty
    } else {
      val size = if (filled) this.size else index
      var extracted = Set[Experience[F]]()
      while (extracted.size < batchSize)
        extracted += array(random.nextInt(size))
      extracted.toSeq
    }
  }
}
object ReplayBuffer {
  case class Experience[F[_]](stateT: F[AgentState], actionT: F[Int], rewardTPlus: F[Double], stateTPlus: F[AgentState])
}
