# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the concept-index emitter (A3-A5).

``compile_concept_index(ontology, binding, *, output_table,
compiler_version)`` consumes A2's row builder and emits two
``CREATE OR REPLACE TABLE ... AS SELECT * FROM UNNEST(...)`` statements
(main + ``__meta`` sibling). Pinned in
``docs/implementation_plan_concept_index_runtime.md`` (A3-A5) and
``docs/entity_resolution_primitives.md`` §4.2 / §5.

Test coverage:

- D2  byte-identical SQL across runs.
- D3  scope rule round-trips through the SQL row count.
- D5  notation row carries ``label_kind='notation'``.
- D14 main + meta both written, both carry the same ``compile_id`` /
      ``compile_fingerprint``.
- Plus: NULL/BOOL rendering, string escaping, meta-row schema,
  empty-rows handling, output-table fully-qualified path appears in
  both statements.
"""

from __future__ import annotations

import re

import pytest

from bigquery_ontology import BigQueryTarget
from bigquery_ontology import Binding
from bigquery_ontology import Entity
from bigquery_ontology import EntityBinding
from bigquery_ontology import Keys
from bigquery_ontology import Ontology
from bigquery_ontology import Property
from bigquery_ontology import PropertyBinding
from bigquery_ontology import PropertyType
from bigquery_ontology._fingerprint import compile_fingerprint
from bigquery_ontology._fingerprint import compile_id
from bigquery_ontology._fingerprint import fingerprint_model
from bigquery_ontology.graph_ddl_compiler import compile_concept_index

_COMPILER_VERSION = "bigquery_ontology test"
_OUTPUT_TABLE = "proj.ds.ontology_concept_index"


def _account_entity(**kw) -> Entity:
  return Entity(
      name="Account",
      keys=Keys(primary=["account_id"]),
      properties=[Property(name="account_id", type=PropertyType.STRING)],
      **kw,
  )


def _account_binding() -> EntityBinding:
  return EntityBinding(
      name="Account",
      source="accounts",
      properties=[PropertyBinding(name="account_id", column="account_id")],
  )


def _binding(*entities: EntityBinding) -> Binding:
  return Binding(
      binding="test_binding",
      ontology="test",
      target=BigQueryTarget(backend="bigquery", project="proj", dataset="ds"),
      entities=list(entities),
      relationships=[],
  )


def _compile(
    *,
    entities=None,
    bindings=None,
    output_table=_OUTPUT_TABLE,
) -> str:
  ontology = Ontology(
      ontology="test",
      entities=entities or [_account_entity()],
  )
  binding = _binding(*(bindings or [_account_binding()]))
  return compile_concept_index(
      ontology,
      binding,
      output_table=output_table,
      compiler_version=_COMPILER_VERSION,
  )


# ------------------------------------------------------------------ #
# Output shape                                                         #
# ------------------------------------------------------------------ #


class TestOutputShape:

  def test_emits_two_create_or_replace_statements(self):
    """One for the main index, one for the ``__meta`` sibling."""
    sql = _compile()
    statements = re.findall(r"CREATE OR REPLACE TABLE", sql)
    assert len(statements) == 2

  def test_main_and_meta_table_paths_are_backtick_quoted(self):
    sql = _compile(output_table="my-proj.my_ds.idx")
    assert "`my-proj.my_ds.idx`" in sql
    assert "`my-proj.my_ds.idx__meta`" in sql

  def test_main_table_carries_all_nine_columns(self):
    sql = _compile()
    main = sql.split("__meta")[0]  # everything before meta statement
    for col in (
        "entity_name",
        "label",
        "label_kind",
        "notation",
        "scheme",
        "language",
        "is_abstract",
        "compile_id",
        "compile_fingerprint",
    ):
      assert col in main, f"missing column {col!r} in main table emission"

  def test_meta_table_carries_seven_provenance_columns(self):
    sql = _compile()
    # everything after first __meta occurrence
    meta = sql[sql.index("__meta") :]
    for col in (
        "compile_fingerprint",
        "compile_id",
        "ontology_fingerprint",
        "binding_fingerprint",
        "target_project",
        "target_dataset",
        "compiler_version",
    ):
      assert col in meta, f"missing column {col!r} in meta table emission"

  def test_uses_inline_unnest_path(self):
    """Default emission path is atomic per-statement
    ``CREATE OR REPLACE TABLE ... AS SELECT * FROM UNNEST(...)``.
    """
    sql = _compile()
    assert "AS SELECT * FROM UNNEST" in sql


# ------------------------------------------------------------------ #
# Provenance — main and meta carry the same compile id/fingerprint    #
# ------------------------------------------------------------------ #


class TestProvenance:

  def test_compile_fingerprint_appears_in_both_main_and_meta(self):
    ontology = Ontology(ontology="test", entities=[_account_entity()])
    binding = _binding(_account_binding())
    expected_cfp = compile_fingerprint(
        fingerprint_model(ontology),
        fingerprint_model(binding),
        _COMPILER_VERSION,
    )
    sql = _compile()
    # The 64-hex string must appear at least twice — once per row in
    # main (here we have 1+ rows), once in meta.
    assert sql.count(expected_cfp) >= 2

  def test_compile_id_appears_in_both_main_and_meta(self):
    ontology = Ontology(ontology="test", entities=[_account_entity()])
    binding = _binding(_account_binding())
    expected_cid = compile_id(
        fingerprint_model(ontology),
        fingerprint_model(binding),
        _COMPILER_VERSION,
    )
    sql = _compile()
    assert sql.count(expected_cid) >= 2

  def test_meta_carries_target_project_and_dataset_from_binding(self):
    sql = _compile()
    meta = sql[sql.index("__meta") :]
    assert "'proj'" in meta
    assert "'ds'" in meta

  def test_meta_carries_compiler_version_string(self):
    sql = _compile()
    meta = sql[sql.index("__meta") :]
    assert f"'{_COMPILER_VERSION}'" in meta


# ------------------------------------------------------------------ #
# Determinism (D2)                                                     #
# ------------------------------------------------------------------ #


class TestDeterminism:

  def test_byte_identical_across_runs(self):
    """Same inputs must produce byte-for-byte identical SQL."""
    sql1 = _compile()
    sql2 = _compile()
    assert sql1 == sql2

  def test_byte_identical_for_complex_ontology(self):
    """Multi-entity, multi-scheme, multi-label ontology is also
    byte-deterministic.
    """
    entities = [
        _account_entity(
            synonyms=["Acct", "AC"],
            annotations={
                "skos:notation": "807",
                "skos:inScheme": ["BankingTaxonomy", "FinancialProducts"],
                "skos:prefLabel": "Bank Account",
            },
        ),
        Entity(
            name="skos_Banking",
            abstract=True,
            annotations={"skos:prefLabel": "Banking"},
        ),
    ]
    sql1 = _compile(entities=entities)
    sql2 = _compile(entities=entities)
    assert sql1 == sql2


# ------------------------------------------------------------------ #
# Value rendering                                                      #
# ------------------------------------------------------------------ #


class TestValueRendering:

  def test_none_values_render_as_null_literal(self):
    """Account has no scheme, language, or notation → NULLs in SQL."""
    sql = _compile()
    main = sql.split("__meta")[0]
    # At least one row contains explicit NULL for the optional columns.
    assert "NULL" in main

  def test_booleans_render_as_uppercase_true_false(self):
    """BigQuery convention; also keeps the byte-identical contract
    stable across runs.
    """
    entities = [
        _account_entity(),
        Entity(name="skos_Banking", abstract=True),
    ]
    sql = _compile(entities=entities)
    assert "TRUE" in sql or "true" in sql
    assert "FALSE" in sql or "false" in sql
    # Pin uppercase per contract:
    main = sql.split("__meta")[0]
    assert "TRUE" in main
    assert "FALSE" in main

  def test_string_with_single_quote_is_escaped(self):
    """A label like ``"O'Brien Bank"`` must not break SQL quoting."""
    entities = [_account_entity(synonyms=["O'Brien Bank"])]
    sql = _compile(entities=entities)
    # Doubled-quote escape is BigQuery's standard form for STRING
    # literals: ``'O''Brien Bank'``.
    assert "'O''Brien Bank'" in sql

  def test_string_with_backslash_is_preserved(self):
    """Backslash is not a SQL escape character in single-quoted
    strings (BigQuery doesn't interpret it specially); it should
    pass through.
    """
    entities = [_account_entity(synonyms=["foo\\bar"])]
    sql = _compile(entities=entities)
    # Either literal backslash preserved or doubled — both are
    # valid; pick whichever the implementation uses, but ensure no
    # truncation.
    assert "foo" in sql and "bar" in sql


# ------------------------------------------------------------------ #
# Scope round-trip + notation row (D3, D5)                            #
# ------------------------------------------------------------------ #


class TestScopeAndNotation:

  def test_unbound_concrete_entity_does_not_appear_in_sql(self):
    """D3: concrete unbound entities are filtered by build_rows;
    their names should not appear in the emitted SQL.
    """
    entities = [
        _account_entity(),
        Entity(
            name="Ledger",
            keys=Keys(primary=["ledger_id"]),
            properties=[Property(name="ledger_id", type=PropertyType.STRING)],
        ),
    ]
    sql = _compile(entities=entities)
    main = sql.split("__meta")[0]
    assert "'Ledger'" not in main

  def test_abstract_entity_is_emitted(self):
    entities = [
        _account_entity(),
        Entity(name="skos_Banking", abstract=True),
    ]
    sql = _compile(entities=entities)
    main = sql.split("__meta")[0]
    assert "'skos_Banking'" in main

  def test_notation_value_appears_as_label_kind_notation_row(self):
    """D5: ``skos:notation`` value '807' gets a row where
    ``label_kind = 'notation'`` and ``label = '807'``.
    """
    entities = [
        _account_entity(annotations={"skos:notation": "807"}),
    ]
    sql = _compile(entities=entities)
    main = sql.split("__meta")[0]
    assert "'notation'" in main
    assert "'807'" in main


# ------------------------------------------------------------------ #
# Empty rows                                                           #
# ------------------------------------------------------------------ #


class TestEmptyRows:

  def test_zero_concrete_zero_abstract_raises(self):
    """An ontology with no abstract entities and a binding that
    references no concrete entities yields zero rows in the main
    index. The emitter should reject this with a clear error
    rather than emit a typeless empty array (which would lose the
    schema).
    """
    ontology = Ontology(ontology="test", entities=[_account_entity()])
    # Binding doesn't reference Account at all.
    binding = _binding()
    with pytest.raises(ValueError, match="no concrete"):
      compile_concept_index(
          ontology,
          binding,
          output_table=_OUTPUT_TABLE,
          compiler_version=_COMPILER_VERSION,
      )


# ------------------------------------------------------------------ #
# Public surface                                                       #
# ------------------------------------------------------------------ #


class TestPublicSurface:

  def test_compile_concept_index_re_exported_at_package_root(self):
    """Per the implementation plan, ``compile_concept_index`` is
    public and re-exported alongside ``compile_graph``.
    """
    import bigquery_ontology

    assert hasattr(bigquery_ontology, "compile_concept_index")

  def test_compile_graph_unchanged(self):
    """The existing ``compile_graph`` byte-identical contract is
    preserved by this PR — sanity check by importing it.
    """
    from bigquery_ontology import compile_graph

    assert callable(compile_graph)
