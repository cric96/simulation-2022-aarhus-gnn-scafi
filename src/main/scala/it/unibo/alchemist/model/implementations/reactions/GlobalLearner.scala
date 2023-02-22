package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist
import it.unibo.learning.Box
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions._
import it.unibo.learning.agents.Learner
import it.unibo.learning.network.torch.writer
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.{IterableHasAsScala, IteratorHasAsScala, MapHasAsScala}

class GlobalLearner[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    random: RandomGenerator,
    layerMolecule: Molecule,
    learner: Learner,
    bufferSize: Int,
    windowSize: Int,
    batchSize: Int,
    actionSpace: ActionSpace.Space,
    episodeLength: Int,
    box: Box
) extends AbstractGlobalReaction[T, P](environment, timeDistribution) {
  private val randomScala = new ScafiIncarnationForAlchemist.AlchemistRandomWrapper(random)

  private val historyMolecule = "history"
  private val buffer = new ReplayBuffer(bufferSize, randomScala)
  lazy val densityMap = environment.getLayer(layerMolecule).get().asInstanceOf[DensityMap[P]]

  private var decayable = List.empty[(String, DecayReference[Any])]

  def attachDecayable(decay: (String, DecayReference[Any])*): Unit = decayable = (decay.toList) ::: decayable

  learner.injectCentralAgent(this)

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val currentTime = this.environment.getSimulation.getTime.toDouble.toInt
    val observationT = computeObservation()
    val stateT = computeStateFromObs(observationT)
    learner.policy
    val actionT = computeAction(stateT)
    performAction(actionT)
    appendHistory(observationT)
    val observationTPlus = computeObservation()
    val stateTPlus = computeStateFromObs(observationTPlus)
    val rewardTPlus = computeRewards()
    writer.add_scalar("Reward", rewardTPlus.values.sum, currentTime)
    stateT.foreach { case (id, stateT) =>
      buffer.put(stateT, actionT(id)._1, rewardTPlus(id), stateTPlus(id))
    }
    learner.update(buffer.sample(batchSize))
    if (currentTime != 0 && (currentTime % episodeLength) == 0) {
      decayable.foreach(_._2.update())
      val newPosition = new Grid(
        environment,
        random,
        0.0f,
        0.0f,
        box.width,
        box.width,
        box.step,
        box.step,
        box.randomness,
        box.randomness
      ).asScala
      newPosition.zip(agents).foreach { case (position, node) =>
        moveNode(position.asInstanceOf[P], node)
      }
    }
  }

  def computeObservation(): Map[Int, Map[Int, NeighborInfo]] = {
    managers
      .map(node => node.node.getId -> node)
      .map { case (id, node) => (id, node, environment.getNeighborhood(node.node).iterator().asScala.toList) }
      .map { case (id, node, neigh) => (id, node, neigh.map(computeInformationFromNeighbour(node.node, _))) }
      .map { case (id, node, neigh) => (id, (computeInformationFromNeighbour(node.node, node.node) :: neigh).toMap) }
      .toMap
  }

  def computeStateFromObs(observations: Map[Int, Map[Int, NeighborInfo]]): Map[Int, AgentState] =
    managers
      .map(node => node.node.getId -> node.get[List[Map[Int, NeighborInfo]]](historyMolecule))
      .map { case (id, history) => (id, AgentState(id, (observations(id) :: history).take(windowSize), ())) }
      .toMap

  def computeAction(states: Map[Int, AgentState]): Map[Int, (Int, Contextual)] =
    states.map { case (id, state) => id -> learner.policy(state) }

  def performAction(actionsIndex: Map[Int, (Int, Contextual)]): Unit = {
    val actions = actionsIndex.map { case (id, (index, _)) => id -> actionSpace(index) }
    actions
      .map { case (id, (angle, module)) =>
        (environment.getNodeByID(id), (angle, module))
      }
      .foreach { case (node, (angle, module)) =>
        node.setConcentration(new SimpleMolecule("angle"), angle.asInstanceOf[T])
        node.setConcentration(new SimpleMolecule("intensity"), module.asInstanceOf[T])
      }
    val deltaVector = actions.map { case (id, (angle, module)) =>
      id -> (module * math.cos(angle), module * math.sin(angle))
    }
    deltaVector
      .map { case (id, movement) => environment.getNodeByID(id) -> movement }
      .foreach { case (node, (dx, dy)) =>
        environment.moveNodeToPosition(node, environment.getPosition(node).plus(Array(dx * 10, -dy * 10)))
      }
  }

  def appendHistory(observations: Map[Int, Map[Int, NeighborInfo]]): Unit =
    managers.foreach { node =>
      node.put(
        historyMolecule,
        (observations(node.node.getId) :: node.get[List[Map[Int, NeighborInfo]]](historyMolecule)).take(windowSize)
      )
      node.put("current", observations(node.node.getId))
    }

  def computeRewards(): Map[Int, Double] = {
    agents.map { node =>
      val neighborhood = environment.getNeighborhood(node).iterator().asScala.toList // ++ List(node)
      /*val densityNeighborhood = neighborhood.map(environment.getPosition).map(densityMap.getValue)
      val probabilityFromNeighborhood = densityNeighborhood.map(_ + 0.001)
      val fixed = probabilityFromNeighborhood.map(d => d / probabilityFromNeighborhood.sum)
      val entropy = fixed.map(p => -p * math.log(p)).sum
      node.getId -> entropy
       */
      val info = neighborhood
        .map(neigh => computeInformationFromNeighbour(node, neigh)._2)
        // compute distance from the vector
        .map { case NeighborInfo(_, (dx, dy), _) => math.sqrt(dx * dx + dy * dy) }
        .maxOption
        .getOrElse(1000.0)
      node.getId -> -info
    }.toMap
  }

  private def computeInformationFromNeighbour(me: Node[T], neighbor: Node[T]): (Int, NeighborInfo) = {
    val myPosition = environment.getPosition(me)
    val neighPosition = environment.getPosition(neighbor)
    val deltaVector = myPosition.minus(neighPosition.getCoordinates).getCoordinates
    neighbor.getId -> NeighborInfo(
      densityMap.getValue(neighPosition),
      (deltaVector(0) / 300, deltaVector(1) / 300),
      -1
    ) // currently I don't consider old action
  }

  private def moveNode(
      position: P,
      node: Node[T]
  ): Unit = {
    initializationComplete(environment.getSimulation.getTime, environment)
    environment.moveNodeToPosition(node, position)
  }

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit =
    managers.foreach(node => node.put(historyMolecule, List.empty[Map[Int, NeighborInfo]]))

  implicit class RichMap[K, V](map: Map[K, V]) {
    def mergeMap(other: Map[K, V]*): Map[K, Seq[V]] =
      map.map { case (k, v) => k -> (Seq(v) ++ other.map(map => map(k))) }
  }
}
