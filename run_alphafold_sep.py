# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by yamule 
# Modified codes (including codes in the subdirectories) are also provided under
# Apache License, Version 2.0

"""Full AlphaFold protein structure prediction script with precomputed a3m."""
# Example command: (Please change --data_dir arg)
# Monomer prediction
# python run_alphafold_sep.py --a3m_list example_files/T1065s1.a3m_5.a3m --model_preset sep --output_dir example_files/results_T1065s1 --no_templates --model_names model_1,model_2 --data_dir /home/ubuntu7/data/disk0/alphafold_check/alphafold_params_2021-10-27/ --hhblits_binary_path none --hhsearch_binary_path none --hmmbuild_binary_path none --hmmsearch_binary_path none --jackhmmer_binary_path none --kalign_binary_path none
# Multimer prediction (The first filename is used for the output directory name.)
# python run_alphafold_sep.py --a3m_list example_files/T1065s2.a3m_5.a3m,example_files/T1065s1.a3m_5.a3m --model_preset sep --output_dir example_files/results_H1065 --no_templates --model_names model_1_multimer,model_2_multimer --data_dir /home/ubuntu7/data/disk0/alphafold_check/alphafold_params_2021-10-27/ --hhblits_binary_path none --hhsearch_binary_path none --hmmbuild_binary_path none --hmmsearch_binary_path none --jackhmmer_binary_path none --kalign_binary_path none 
# Please change /home/ubuntu7/data/disk0/alphafold_check/alphafold_params_2021-10-27/ to the path of the directory which you downloaded the AF2 weight files.

import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmbuild
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data import msa_pairing;
import numpy as np

import gzip;
import copy;

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)
flags.DEFINE_list('a3m_list', None, 'Paths to a3m files. If multiple paths are provided, the run is considered as complex prediction.')
flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'none',
                  ['full_dbs', 'reduced_dbs','none'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs) or'
                  'do not check (none)')
flags.DEFINE_enum('model_preset', 'sep',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer','sep'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model or set explicitly with --model_names ')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('run_relax', True, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('use_gpu_relax', True, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_boolean('save_prevs', False, 'Save results of each recycling step.')
flags.DEFINE_boolean('gzip_features', False, 'Treat feature pickles with gzipped.')


flags.DEFINE_boolean('keep_unpaired', False, 'Possibly avoid homo-multimer clash problem. https://twitter.com/sokrypton/status/1457639018141728770'
'Paired sequences are created, too.'
'Possibly unpaired sequences are filtered out by some steps which I couldn\'t follow. (yamule)'
);
flags.DEFINE_boolean('no_templates', False, 'Do not use templates.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    save_prevs:bool = False,
    gzip_features:bool = False):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
    input_fasta_path=fasta_path,
    msa_output_dir=msa_output_dir)
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if gzip_features:
    with gzip.open(features_output_path+'.gz', 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  else:
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    if model_runner.save_prevs:
      # Save results of each recycling step.
      prevs = prediction_result['prevs'];
      pnum = prevs['pos'].shape[0];

      dummybuff = copy.deepcopy(prediction_result);

      for pp in range(pnum):
        out_pdb_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.pdb');
        out_pkl_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.metrics.pkl');
        dummybuff['structure_module']['final_atom_positions'] = prevs['pos'][pp];
        if 'predicted_aligned_error' in dummybuff:
          dummybuff['predicted_aligned_error'] = {};
          dummybuff['predicted_aligned_error']['logits'] = prevs['predicted_aligned_error_logits'][pp];
          dummybuff['predicted_aligned_error']['breaks'] = dummybuff['predicted_aligned_error_breaks'];
          dummybuff['predicted_aligned_error']['asym_id'] = processed_feature_dict['asym_id'];
        dummybuff['predicted_lddt']['logits'] = prevs['predicted_lddt_logits'][pp];

        cres = model.get_confidence_metrics(dummybuff, multimer_mode=model_runner.multimer_mode);

        out_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=dummybuff,
        b_factors= np.repeat(cres['plddt'][:, None], residue_constants.atom_type_num, axis=-1),
        remove_leading_feature_dimension=not model_runner.multimer_mode);
        if gzip_features:
          with gzip.open(out_pkl_path+'.gz','wb') as f:
            pickle.dump(cres,f,protocol=4);
        else:
          with open(out_pkl_path,'wb') as f:
            pickle.dump(cres,f,protocol=4);

        with open(out_pdb_path, 'w') as f:
          f.write(protein.to_pdb(out_protein));
        
        del cres;
        del out_protein;
      if 'predicted_aligned_error_breaks' in prediction_result:
        del prediction_result['predicted_aligned_error_breaks']
      del prevs;
      del prediction_result['prevs'];
      del dummybuff;

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    if gzip_features:
      with gzip.open(result_output_path+'.gz', 'wb') as f:
        pickle.dump(prediction_result, f, protocol=4)
    else:
      with open(result_output_path, 'wb') as f:
        pickle.dump(prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    if amber_relaxer:
      # Relax the prediction.
      t_0 = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      timings[f'relax_{model_name}'] = time.time() - t_0

      relaxed_pdbs[model_name] = relaxed_pdb_str

      # Save the relaxed PDB.
      relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  run_multimer_system = 'multimer' in FLAGS.model_preset
  
  if FLAGS.model_preset == 'sep':
    for mm in list(FLAGS.model_names):
      if "multimer" in mm:
        run_multimer_system = True;

  if not FLAGS.db_preset in ( 'none',): 
    _check_flag('small_bfd_database_path', 'db_preset',
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset',
                should_be_set=not use_small_bfd)
    _check_flag('uniclust30_database_path', 'db_preset',
                should_be_set=not use_small_bfd)
    _check_flag('pdb70_database_path', 'model_preset',
                should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
                should_be_set=run_multimer_system)
  
  msa_pairing.KEEP_UNPAIRED = FLAGS.keep_unpaired;

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.a3m_list]
  if not run_multimer_system:
    if len(fasta_names) != len(set(fasta_names)):
      raise ValueError('All FASTA paths must have a unique basename.')

  if not FLAGS.no_templates:
    if run_multimer_system:
      template_searcher = hmmsearch.Hmmsearch(
          binary_path=FLAGS.hmmsearch_binary_path,
          hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
          database_path=FLAGS.pdb_seqres_database_path)
      template_featurizer = templates.HmmsearchHitFeaturizer(
          mmcif_dir=FLAGS.template_mmcif_dir,
          max_template_date=FLAGS.max_template_date,
          max_hits=MAX_TEMPLATE_HITS,
          kalign_binary_path=FLAGS.kalign_binary_path,
          release_dates_path=None,
          obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
    else:
      template_searcher = hhsearch.HHSearch(
          binary_path=FLAGS.hhsearch_binary_path,
          databases=[FLAGS.pdb70_database_path])
      template_featurizer = templates.HhsearchHitFeaturizer(
          mmcif_dir=FLAGS.template_mmcif_dir,
          max_template_date=FLAGS.max_template_date,
          max_hits=MAX_TEMPLATE_HITS,
          kalign_binary_path=FLAGS.kalign_binary_path,
          release_dates_path=None,
          obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = None;
    template_featurizer = None;

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas
      ,use_a3m = True
      ,search_templates=not FLAGS.no_templates
      ,for_multimer=run_multimer_system)

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  if FLAGS.model_preset != 'sep':
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  else:
    model_names = FLAGS.model_names;
  for model_name in model_names:
    print(model_name)
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params,save_prevs=FLAGS.save_prevs)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  if FLAGS.run_relax:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax)
  else:
    amber_relaxer = None

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  if run_multimer_system:
    fasta_name = fasta_names[0]
    predict_structure(
        fasta_path=",".join(FLAGS.a3m_list),
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        save_prevs=FLAGS.save_prevs,
        gzip_features=FLAGS.gzip_features
        );
  else:
  # Predict structure for each of the sequences.
    for i,(fasta_path, fasta_name) in enumerate(zip(FLAGS.a3m_list, fasta_names)):
      fasta_name = fasta_names[i]
      predict_structure(
          fasta_path=fasta_path,
          fasta_name=fasta_name,
          output_dir_base=FLAGS.output_dir,
          data_pipeline=data_pipeline,
          model_runners=model_runners,
          amber_relaxer=amber_relaxer,
          benchmark=FLAGS.benchmark,
          random_seed=random_seed,
          gzip_features=FLAGS.gzip_features)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'a3m_list',
      'output_dir',
      #'model_names',
      'data_dir',
      #'uniref90_database_path',
      #'mgnify_database_path',
      #'template_mmcif_dir',
      #'max_template_date',
      #'obsolete_pdbs_path',
  ])

  app.run(main)
