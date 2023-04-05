package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.GraphReplayBuffer._
import it.unibo.learning.network.Graph

import scala.collection.mutable
import scala.util.Random

trait GraphReplayBuffer {
  val random: Random
  def put(
      stateT: Graph[AgentState],
      actionT: Graph[Int],
      rewardTplus: Graph[Double],
      stateTPlus: Graph[AgentState]
  ): Unit
  def sample(batchSize: Int): Seq[GraphExperience]
}

class GraphArrayReplayBuffer(size: Int, val random: Random) {
  private val array = Array.ofDim[GraphExperience](size)
  private var index = 0
  private var filled = false
  def put(
      stateT: Graph[AgentState],
      actionT: Graph[Int],
      rewardTplus: Graph[Double],
      stateTPlus: Graph[AgentState]
  ): Unit = {
    array(index) = GraphExperience(stateT, actionT, rewardTplus, stateTPlus)
    if (index == size - 1)
      filled = true
    index = (index + 1) % size
  }

  def sample(batchSize: Int): Seq[GraphExperience] = {
    if (batchSize > index && !filled) {
      Seq.empty
    } else {
      val size = if (filled) this.size else index
      var extracted = Set[GraphExperience]()
      while (extracted.size < batchSize)
        extracted += array(random.nextInt(size))
      extracted.toSeq
    }
  }

}
object GraphReplayBuffer {
  case class GraphExperience(
      stateT: Graph[AgentState],
      actionT: Graph[Int],
      rewardTPlus: Graph[Double],
      stateTPlus: Graph[AgentState]
  )
}
