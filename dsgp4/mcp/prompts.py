"""
MCP prompts of the dSGP4 server: reusable workflow templates that guide a model
through the typical multi-tool analyses the library supports.
"""


def register(server):
    """Registers the prompts on the given `MCPServer`."""

    @server.prompt(description='Full analysis of a single element set: validation, orbital elements, '
                               'derived quantities, propagation and a rendered orbit plot.')
    def characterize_orbit(element_set: str) -> str:
        return """\
Analyze the following element set (a TLE or a CCSDS OMM message) with the dsgp4 tools:

{element_set}

Proceed as follows:
1. If it is a TLE, check it with `validate_tle` and report any format or checksum problem.
2. Parse it with `parse_element_set` and present the orbital elements and the derived
   quantities (semi-major axis, period, perigee/apogee altitude, orbit class, epoch, and
   whether deep-space corrections apply).
3. Propagate one full orbital period with `propagate` (a handful of evenly spaced points is
   enough) and comment on the trajectory.
4. Render the orbit with `plot_orbits`.
5. Close with a concise summary of what kind of object/orbit this is, mentioning that SGP4
   states are in the TEME frame and that TLE accuracy is on the order of km.

If any tool reports warnings (e.g. a decayed satellite), surface them prominently.
""".format(element_set=element_set)

    @server.prompt(description='Compare two element sets: elements side by side, geometry differences '
                               'and, when meaningful, the relative distance over a common time window.')
    def compare_orbits(element_set_a: str, element_set_b: str) -> str:
        return """\
Compare the two following element sets with the dsgp4 tools.

Element set A:
{element_set_a}

Element set B:
{element_set_b}

Proceed as follows:
1. Parse both with `parse_element_set` and build a side-by-side table of the elements and
   derived quantities (period, perigee/apogee altitude, inclination, orbit class, epoch).
2. Discuss the geometric differences (altitude regimes, planes, eccentricity) and what they
   imply for the two objects.
3. If the epochs are close enough for a meaningful comparison, choose a short common time
   window, propagate both to the same UTC dates with `propagate` (use `dates_utc`) and
   report the relative distance at each date.
4. Plot both orbits in a single figure with `plot_orbits` (pass the two element sets in one
   string, separated by a newline).
5. Summarize the comparison, being explicit about the km-level accuracy of SGP4 and about
   the epochs of the two element sets.
""".format(element_set_a=element_set_a, element_set_b=element_set_b)

    @server.prompt(description='Uncertainty workflow: state transition matrices via autodiff and the '
                               'mapping of a TLE-space covariance into a Cartesian TEME covariance.')
    def uncertainty_analysis(element_set: str) -> str:
        return """\
Perform a differentiability/uncertainty analysis of the following element set with the
dsgp4 tools:

{element_set}

Proceed as follows:
1. Read the `dsgp4://reference/sgp4-parameters` resource to recall the order and the
   internal units of the nine differentiable SGP4 parameters.
2. Compute the Jacobian of the state at a few times (e.g. 0, half a period, one period)
   with `state_partials_wrt_tle` and identify which parameters the state is most sensitive
   to, and how the sensitivities grow with time (e.g. the along-track growth driven by the
   mean motion and mean anomaly columns).
3. Compute `state_partials_wrt_time` at the same times and check the position derivative
   against the propagated velocity (they should match once km/min is converted to km/s).
4. If the user has a covariance of the TLE parameters, map it with `transform_covariance`;
   otherwise illustrate the workflow with a small diagonal example covariance on the six
   mean elements, and report the resulting position/velocity sigmas.
5. Summarize what the gradients say about the orbit determination/uncertainty of this
   object, quoting the units of every number.
""".format(element_set=element_set)

    @server.prompt(description='Gradient-based TLE determination: re-epoch a TLE or fit one to a '
                               'Cartesian state with the differentiable Newton-Raphson tools.')
    def tle_determination(element_set: str) -> str:
        return """\
The user needs a new TLE consistent with a target trajectory or state, starting from the
following element set:

{element_set}

Choose the right dsgp4 estimation tool:
- To move the element set to a new epoch (same trajectory, new reference time), call
  `update_tle_epoch` with the desired ISO UTC date.
- To find the TLE matching a known Cartesian TEME state (position km, velocity km/s) at a
  given date, call `fit_tle_to_state`, using the element set above as the template/initial
  guess.

In both cases:
1. Report the fitted TLE lines and its parsed elements.
2. Check the reported fit residuals: position residuals should be well below a km for a
   converged fit; if they are large, say so clearly and consider more iterations or a
   closer template orbit.
3. Optionally verify by propagating the fitted TLE with `propagate` and comparing against
   the target.
Remember that the fitted elements are SGP4 *mean* elements: they only make sense together
with the SGP4 model, and the B* / drag fields are inherited from the template.
""".format(element_set=element_set)

    @server.prompt(description='How to train the hybrid ML-dSGP4 model on higher-precision ephemerides '
                               '(a Python workflow using the library directly).')
    def mldsgp4_training_guide() -> str:
        return """\
The user wants to train the hybrid ML-dSGP4 model (`dsgp4.mldsgp4`) so that neural
networks correct SGP4 towards higher-precision data. Training is a Python workflow that
uses the library directly (the MCP tools only run already-trained checkpoints); guide the
user through writing it:

1. Read the `dsgp4://reference/mldsgp4` resource for the architecture.
2. Data: pairs of (element set, time since epoch) inputs and higher-precision TEME states
   (numerical propagation or precise ephemerides) as targets. Normalize the targets with
   the same `normalization_R`/`normalization_V` constants the model uses.
3. Sketch of the training loop:

```python
import torch, dsgp4
model = dsgp4.mldsgp4(hidden_size=100)          # differentiable end to end
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    for tles, tsinces, targets_normalized in loader:   # your dataset
        optimizer.zero_grad()
        out = model(tles, tsinces)                     # normalized 6-vector states
        loss = torch.nn.functional.mse_loss(out, targets_normalized)
        loss.backward()                                # gradients flow through SGP4
        optimizer.step()
torch.save(model.state_dict(), 'mldsgp4.pth')
```

4. Practical notes: batch TLEs with one time per TLE (the batched propagator's
   convention); keep `hidden_size` and the correction scales consistent between training
   and inference; validate on held-out epochs, monitoring the position error in km
   (multiply the normalized error by `normalization_R`).
5. Once a checkpoint exists, it can be evaluated through the MCP `mldsgp4_propagate` tool
   (passing `model_path`) and inspected with `mldsgp4_inspect`.
"""
