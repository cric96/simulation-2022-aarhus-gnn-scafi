package it.unibo.learning.network

import it.unibo.learning.network.torch.geometric
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from

object GNNDQN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic = {
    val module = geometric.nn.Sequential(
      "x, edge_index",
      Seq(
        // (geometric.nn.LEConv(input, hidden), "x, edge_index -> x".as[py.Any]).as[py.Any],
        // (geometric.nn.GATv2Conv(input, hidden), "x, edge_index -> x".as[py.Any]).as[py.Any],
        // (geometric.nn.SAGEConv(input, hidden), "x, edge_index -> x".as[py.Any]).as[py.Any],
        // nn.SuperGATConv
        (geometric.nn.SuperGATConv(input, hidden), "x, edge_index -> x".as[py.Any]).as[py.Any],
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, output)
      ).toPythonCopy
    )
    module
  }
}
