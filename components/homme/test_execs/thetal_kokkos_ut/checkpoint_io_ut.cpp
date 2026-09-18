/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

// Round-trip test for CheckpointStore (HOMME adjoint battleplan, Step 3):
// save a StateSnapshot to disk, reload it, and confirm the reload reproduces
// the original exactly. Also exercises the "resume forward integration" use
// case the backward sweep will actually rely on: a checkpoint taken mid-way
// through a trajectory must stay unaffected by whatever happens to the live
// state afterwards.
//
// No F90 sources needed (a StateSnapshot is just a bundle of Kokkos views),
// but CheckpointStore's own I/O (PIOc_write_darray/read_darray, run through
// a real decomposition) is collective over its communicator -- harmless
// with the single rank this test runs with (the collectives just degenerate
// to the trivial case), but exercised for real nonetheless.

#include <catch2/catch.hpp>

#include <random>
#include <string>

#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>

#include "CheckpointIO.hpp"
#include "StateSnapshot.hpp"
#include "Types.hpp"

#include "utilities/TestUtils.hpp"

#include <ekat_comm.hpp>
#include <ekat_pack_kokkos.hpp>

using namespace Homme;

namespace {

template<typename rngAlg, typename PDF>
void randomize_state_snapshot (StateSnapshot& s, rngAlg& engine, PDF&& pdf) {
  genRandArray(s.v,        engine, pdf);
  genRandArray(s.vtheta_dp,engine, pdf);
  genRandArray(s.dp3d,     engine, pdf);
  genRandArray(s.w_i,      engine, pdf);
  genRandArray(s.phinh_i,  engine, pdf);
  if (s.ps_v.data()!=nullptr) {
    genRandArray(s.ps_v,engine, pdf);
  }
}

// Creates (if needed) and returns a scratch directory private to this test
// process, for CheckpointStore's files to live in.
std::string scratch_dir () {
  std::string dir = "/tmp/homme_checkpoint_io_ut_" + std::to_string(getpid());
  mkdir(dir.c_str(),0755);
  return dir;
}

} // anonymous namespace

TEST_CASE("checkpoint_io_roundtrip", "checkpoint_io") {
  using rngAlg = std::mt19937_64;
  using dpdf = std::uniform_real_distribution<double>;

  std::random_device rd;
  constexpr int num_elems = 3;
  const unsigned int catchRngSeed = Catch::rngSeed();
  const unsigned int seed = catchRngSeed==0 ? rd() : catchRngSeed;
  std::cout << "seed: " << seed << (catchRngSeed==0 ? " (catch rng seed was 0)\n" : "\n");
  rngAlg engine(seed);

  ekat::Comm comm(MPI_COMM_WORLD);
  CheckpointStore store(scratch_dir(),comm,num_elems);

  SECTION ("round trip, with ps") {
    StateSnapshot orig(num_elems,/*alloc_ps=*/true);
    randomize_state_snapshot(orig,engine,dpdf(-1,1));

    const CheckpointId id{0,1,2};
    REQUIRE_FALSE(store.has(id));
    store.save(id,orig);
    REQUIRE(store.has(id));

    StateSnapshot loaded(num_elems,/*alloc_ps=*/true);
    loaded.zero();
    store.load(id,loaded);

    REQUIRE(views_are_equal(ekat::scalarize(orig.v),        ekat::scalarize(loaded.v),        NUM_PHYSICAL_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(orig.vtheta_dp),ekat::scalarize(loaded.vtheta_dp),NUM_PHYSICAL_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(orig.dp3d),     ekat::scalarize(loaded.dp3d),     NUM_PHYSICAL_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(orig.w_i),      ekat::scalarize(loaded.w_i),      NUM_INTERFACE_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(orig.phinh_i),  ekat::scalarize(loaded.phinh_i),  NUM_INTERFACE_LEV));
    REQUIRE(views_are_equal(orig.ps_v,loaded.ps_v,NP));

    store.discard(id.nn_call);
    REQUIRE_FALSE(store.has(id));
  }

  SECTION ("round trip, without ps") {
    // ps_v is optional (see StateSnapshot's alloc_ps flag); the checkpoint
    // file's "has_ps" attribute must round-trip that choice too.
    StateSnapshot orig(num_elems,/*alloc_ps=*/false);
    randomize_state_snapshot(orig,engine,dpdf(-1,1));

    const CheckpointId id{1,0,0};
    store.save(id,orig);

    StateSnapshot loaded(num_elems,/*alloc_ps=*/false);
    loaded.zero();
    store.load(id,loaded);

    REQUIRE(views_are_equal(ekat::scalarize(orig.v),ekat::scalarize(loaded.v),NUM_PHYSICAL_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(orig.phinh_i),ekat::scalarize(loaded.phinh_i),NUM_INTERFACE_LEV));

    store.discard(id.nn_call);
  }

  SECTION ("multiple snapshots share one file") {
    // Successive checkpoints within the same nn_call must land in the same
    // file (as distinct records), not clobber one another, and be
    // independently reloadable regardless of the order they're loaded back
    // in -- the backward sweep reads them in the reverse of save() order.
    constexpr int nn_call = 3;
    StateSnapshot snap_a(num_elems,/*alloc_ps=*/true);
    StateSnapshot snap_b(num_elems,/*alloc_ps=*/true);
    StateSnapshot snap_c(num_elems,/*alloc_ps=*/true);
    randomize_state_snapshot(snap_a,engine,dpdf(-1,1));
    randomize_state_snapshot(snap_b,engine,dpdf(-1,1));
    randomize_state_snapshot(snap_c,engine,dpdf(-1,1));

    const CheckpointId id_a{nn_call,0,0};
    const CheckpointId id_b{nn_call,0,1};
    const CheckpointId id_c{nn_call,1,0};
    store.save(id_a,snap_a);
    store.save(id_b,snap_b);
    store.save(id_c,snap_c);

    StateSnapshot loaded(num_elems,/*alloc_ps=*/true);

    // Load in reverse order, on purpose.
    loaded.zero();
    store.load(id_c,loaded);
    REQUIRE(views_are_equal(ekat::scalarize(snap_c.v),ekat::scalarize(loaded.v),NUM_PHYSICAL_LEV));

    loaded.zero();
    store.load(id_a,loaded);
    REQUIRE(views_are_equal(ekat::scalarize(snap_a.v),ekat::scalarize(loaded.v),NUM_PHYSICAL_LEV));

    loaded.zero();
    store.load(id_b,loaded);
    REQUIRE(views_are_equal(ekat::scalarize(snap_b.v),ekat::scalarize(loaded.v),NUM_PHYSICAL_LEV));

    store.discard(nn_call);
  }

  SECTION ("resume forward integration") {
    // Mimics the backward sweep's actual use case: checkpoint an
    // intermediate StateSnapshot, keep "integrating forward" (here, just
    // perturb the live copy further), then reload the checkpoint and
    // confirm it still matches the state as it was *at checkpoint time* --
    // i.e. it is unaffected by what happens to the live state afterwards.
    StateSnapshot live(num_elems,/*alloc_ps=*/true);
    randomize_state_snapshot(live,engine,dpdf(-1,1));

    StateSnapshot at_checkpoint = live.clone(true);

    const CheckpointId id{2,3,4};
    store.save(id,live);

    StateSnapshot delta(num_elems,true);
    randomize_state_snapshot(delta,engine,dpdf(-1,1));
    live.add(delta);

    StateSnapshot reloaded(num_elems,/*alloc_ps=*/true);
    reloaded.zero();
    store.load(id,reloaded);

    REQUIRE(views_are_equal(ekat::scalarize(at_checkpoint.v),        ekat::scalarize(reloaded.v),        NUM_PHYSICAL_LEV));
    REQUIRE(views_are_equal(ekat::scalarize(at_checkpoint.vtheta_dp),ekat::scalarize(reloaded.vtheta_dp),NUM_PHYSICAL_LEV));

    // The live state kept moving after the checkpoint was taken, so it
    // should now differ from what was (and still is) on disk.
    REQUIRE_FALSE(views_are_equal(ekat::scalarize(live.v),ekat::scalarize(reloaded.v),NUM_PHYSICAL_LEV));

    store.discard(id.nn_call);
  }
}
