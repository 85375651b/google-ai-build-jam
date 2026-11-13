"""Real SQLite transactions, author evidence, naming and human-readable projections."""
import json
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from backend.database import SessionLocal, Base
from backend.main import app
from backend.models import (Entity, EntityProperty, EntityName, EntityRelationship, EntityRevision,
    CanonOperationBatch, CanonSyncIssue, Fact, FactRevision, BibleSection, Conversation, World)
from backend.schemas import EntityOperation as Op, WorldCreate, BibleSectionUpdate, TurnAnalysisSchema, CreativePermissionSchema, AuthorDecisionSchema
from backend.services.entity_service import entity_service
from backend.services.bible_service import bible_service
from backend.services.world_service import world_service
from backend.services.author_intent import reconcile_turn
from backend.services.context_retrieval import context_retriever
from backend.services.llm_service import LocalModelError


@pytest.fixture
def project():
    with SessionLocal() as db:
        world = world_service.create_world(db, WorldCreate(name='Structured test'))
        yield db, world
        db.rollback(); world_service.delete_world(db, db.get(World, world.id))


def apply(project, *operations, request_id=None):
    db, world = project
    for op in operations:
        entity = entity_service.resolve(db, world.id, op.entity_id, op.entity_ref, required=False)
        if entity and op.expected_version is None:
            op.expected_version = entity.version_number
        if op.related_entity_ref or op.related_entity_id:
            target = entity_service.resolve(db, world.id, op.related_entity_id, op.related_entity_ref, required=False)
            if target and op.related_expected_version is None:
                op.related_expected_version = target.version_number
    return entity_service.execute(db, world.id, list(operations), request_id or str(uuid4()), manual=True)


def rename(ref='Planet X', value='Pallene', **kwargs):
    entity_type = kwargs.pop('entity_type', ref.split()[0].lower())
    return Op(action='rename', entity_ref=ref, entity_type=entity_type, value=value, **kwargs)


def get_entity(project, ref='Planet X'):
    db, world = project
    return entity_service.resolve(db, world.id, ref=ref)


def value(project, prop='canonical_name', ref='Planet X'):
    db, _ = project
    return json.loads(db.query(EntityProperty).filter_by(entity_id=get_entity(project, ref).id, property_key=prop).one().value_json)


def test_identity_rename_alias_former_and_fact_dedup(project):
    apply(project, rename())
    db, world = project
    entity_id = get_entity(project).id
    fact_id = db.query(EntityProperty).filter_by(entity_id=entity_id).one().fact_id
    apply(project, Op(action='add_alias', entity_ref='Planet X', value='The Pearl'))
    assert value(project) == 'Pallene'
    apply(project, rename(value='Neso'))
    assert get_entity(project).id == entity_id
    assert entity_service.resolve(db, world.id, ref='The Pearl').id == entity_id
    assert db.query(EntityName).filter_by(entity_id=entity_id, kind='FORMER', name='Pallene').count() == 1
    assert db.query(Fact).filter_by(world_id=world.id).count() == 1
    assert db.query(EntityProperty).filter_by(entity_id=entity_id).one().fact_id == fact_id
    assert db.query(EntityRevision).filter_by(entity_id=entity_id).count() == 3
    assert db.query(FactRevision).filter_by(fact_id=fact_id).count() == 2


def test_reservation_scopes_and_alias_overlap(project):
    apply(project, Op(action='reserve_name', entity_ref='Planet Y', entity_type='planet', value='Enceladus'))
    with pytest.raises(LocalModelError, match='reserved'):
        apply(project, rename(value='Enceladus'))
    assert get_entity(project, 'Planet Y')
    apply(project, rename(ref='City A', value='Enceladus'))
    apply(project, rename(ref='Character A', value='Enceladus', scope_key='culture-a'))
    apply(project, rename(ref='Character B', value='Enceladus', scope_key='culture-b'))
    apply(project, Op(action='add_alias', entity_ref='City A', value='Enceladus'))
    with pytest.raises(LocalModelError):
        entity_service.resolve(project[0], project[1].id, ref='Enceladus')


def test_cross_entity_author_correction_atomic_no_replacement(project):
    apply(project, rename(value='Enceladus'), Op(action='create_entity', entity_ref='Planet Y', entity_type='planet'))
    db, world = project
    x_id, y_id = get_entity(project).id, get_entity(project, 'Planet Y').id
    message = "Enceladus is Planet Y's name, not Planet X's."
    analysis = reconcile_turn(message, TurnAnalysisSchema(primary_intent='FACTUAL_RESEARCH',
        author_decisions=[AuthorDecisionSchema(kind='CORRECTION', source_quote=message)],
        creative_permission=CreativePermissionSchema(level='FACTUAL_ONLY')))
    ops = entity_service.compile_author_operations(analysis, message)
    result = entity_service.execute(db, world.id, ops, 'cross-correction', analysis=analysis,
        author_text=message, expected_versions={x_id:get_entity(project).version_number, y_id:get_entity(project,'Planet Y').version_number})
    assert result['status'] == 'committed' and result['clarifications']
    assert value(project) is None and value(project, ref='Planet Y') == 'Enceladus'
    assert get_entity(project).id == x_id and get_entity(project, 'Planet Y').id == y_id
    assert db.query(EntityName).filter_by(entity_id=x_id, kind='REJECTED', name='Enceladus').count() == 1
    assert db.query(Fact).filter_by(world_id=world.id, canon_status='ESTABLISHED_CANON').count() == 1


def test_failed_multi_entity_batch_rolls_back_everything(project):
    apply(project, rename(ref='Planet Z', value='Reserved'))
    db, world = project
    before = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one().markdown_content
    with pytest.raises(LocalModelError):
        apply(project, rename(value='New X'), rename(ref='Planet Y', value='Reserved'))
    assert db.query(Entity).filter_by(world_id=world.id).count() == 1
    assert db.query(Fact).filter_by(world_id=world.id).count() == 1
    assert db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one().markdown_content == before


def test_mid_batch_database_failure_rolls_back(project, monkeypatch):
    db, world = project
    original = entity_service._project
    counter = []
    def broken(*args, **kwargs):
        original(*args, **kwargs); counter.append(1)
        if len(counter) == 2:
            raise RuntimeError('Injected database failure')
    monkeypatch.setattr(entity_service, '_project', broken)
    with pytest.raises(RuntimeError):
        apply(project, rename(), rename(ref='Planet Y', value='Enceladus'))
    assert db.query(Entity).filter_by(world_id=world.id).count() == 0
    assert db.query(Fact).filter_by(world_id=world.id).count() == 0
    assert all(not s.markdown_content for s in db.query(BibleSection).filter_by(world_id=world.id))


def test_retry_is_exactly_once_and_changed_payload_conflicts(project):
    op = rename()
    result = apply(project, op, request_id='retry')
    # Same original payload, including the original version, is replayed verbatim.
    result2 = entity_service.execute(project[0], project[1].id, [op], 'retry', manual=True)
    assert result == result2
    with pytest.raises(LocalModelError) as exc:
        apply(project, rename(value='Different'), request_id='retry')
    assert exc.value.error_code == 'BATCH_ID_CONFLICT'
    assert project[0].query(EntityRevision).count() >= 1


def test_expected_version_rejects_stale_edit(project):
    apply(project, rename())
    version = get_entity(project).version_number
    apply(project, rename(value='Neso'))
    with pytest.raises(LocalModelError) as exc:
        entity_service.execute(project[0], project[1].id, [rename(value='Stale', expected_version=version)], 'stale', manual=True)
    assert exc.value.error_code == 'STALE_ENTITY_VERSION'
    assert value(project) == 'Neso'


def test_parent_chronology_and_atomic_change(project):
    apply(project, Op(action='set_property', entity_ref='Character A', entity_type='character', property_key='birth_year', value=100),
          Op(action='set_property', entity_ref='Character B', entity_type='character', property_key='birth_year', value=130))
    apply(project, Op(action='add_relationship', entity_ref='Character A', related_entity_ref='Character B', relationship_type='parent_of'))
    with pytest.raises(LocalModelError, match='parent'):
        apply(project, Op(action='set_property', entity_ref='Character A', property_key='birth_year', value=150))
    assert value(project, 'birth_year', 'Character A') == 100
    apply(project, Op(action='set_property', entity_ref='Character A', property_key='birth_year', value=150),
          Op(action='set_property', entity_ref='Character B', property_key='birth_year', value=180))
    assert value(project, 'birth_year', 'Character B') == 180


def test_event_participation_chronology(project):
    apply(project, Op(action='set_property', entity_ref='Character A', entity_type='character', property_key='birth_year', value=100),
          Op(action='set_property', entity_ref='Event A', entity_type='event', property_key='year', value=150))
    apply(project, Op(action='add_relationship', entity_ref='Character A', related_entity_ref='Event A', relationship_type='participates_in'))
    with pytest.raises(LocalModelError):
        apply(project, Op(action='set_property', entity_ref='Event A', property_key='year', value=90))
    assert value(project, 'year', 'Event A') == 150


def test_immutable_property_and_relationship_removal(project):
    apply(project, rename())
    apply(project, Op(action='add_constraint', entity_ref='Planet X', constraint_type='immutable_property', value={'property_key':'canonical_name','value':'Pallene'}))
    with pytest.raises(LocalModelError):
        apply(project, rename(value='Neso'))
    apply(project, Op(action='remove_constraint', entity_ref='Planet X', constraint_type='immutable_property', value={'property_key':'canonical_name','value':'Pallene'}), rename(value='Neso'))
    assert value(project) == 'Neso'


def test_manual_markdown_sync_and_ambiguous_edit_preservation(project):
    apply(project, rename())
    db, world = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    original_id = get_entity(project).id
    bible_service.update_section_directly(db, section, BibleSectionUpdate(markdown_content='Intro stays.\n\nPlanet X is called Neso.', expected_version_number=section.version_number))
    assert value(project) == 'Neso' and get_entity(project).id == original_id
    ambiguous = 'Intro stays.\n\nPlanet X might instead have a name I have not chosen.'
    bible_service.update_section_directly(db, section, BibleSectionUpdate(markdown_content=ambiguous, expected_version_number=section.version_number))
    assert section.markdown_content == ambiguous
    assert db.query(CanonSyncIssue).filter_by(world_id=world.id, status='OPEN').count() == 1
    assert get_entity(project).canon_status == 'NEEDS_REVIEW'


def test_agent_update_does_not_overwrite_author_prose(project):
    apply(project, rename())
    db, world = project
    entity_version = get_entity(project).version_number
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    bible_service.update_section_directly(db, section, BibleSectionUpdate(markdown_content='Planet X is called AuthorChoice.', expected_version_number=section.version_number))
    with pytest.raises(LocalModelError):
        entity_service.execute(db, world.id, [rename(value='Old AI', expected_version=entity_version)], 'old-agent', manual=True)
    assert 'AuthorChoice' in db.get(BibleSection, section.id).markdown_content


def test_model_cannot_use_valid_quote_for_unrelated_property(project):
    db, world = project
    message = 'Planet X is called Pallene.'
    analysis = reconcile_turn(message, TurnAnalysisSchema(primary_intent='CANON_DECLARATION', creative_permission=CreativePermissionSchema(level='FACTUAL_ONLY')))
    with pytest.raises(LocalModelError):
        entity_service.execute(db, world.id, [Op(action='set_property', entity_ref='Planet X', entity_type='planet', property_key='history', value='Invented empire', source_quote=message)], 'bad-evidence', analysis=analysis, author_text=message)
    assert not db.query(Entity).filter_by(world_id=world.id).count()


def test_structured_context_retrieves_ids_relationships_and_constraints(project):
    apply(project, rename(), Op(action='reserve_name', entity_ref='Planet Y', entity_type='planet', value='Enceladus'))
    db, world = project
    bundle = context_retriever.retrieve(db, world.id, 'Help me name Planet X.')
    text_value = bundle.render({'structured_entity'})
    assert get_entity(project).id in text_value and get_entity(project, 'Planet Y').id in text_value
    assert 'RESERVED' in text_value and 'Enceladus' in text_value


def test_archive_roundtrip_preserves_graph_provenance_and_revisions(project):
    apply(project, rename(), Op(action='reserve_name', entity_ref='Planet Y', entity_type='planet', value='Enceladus'))
    apply(project, rename(value='Neso'))
    db, world = project
    archive = world_service.export_project_bundle(db, world)
    restored = world_service.import_project_bundle(db, archive)
    try:
        copy = entity_service.resolve(db, restored.id, ref='Planet X')
        assert copy.id != get_entity(project).id
        assert json.loads(db.query(EntityProperty).filter_by(entity_id=copy.id, property_key='canonical_name').one().value_json) == 'Neso'
        assert db.query(EntityRevision).filter_by(entity_id=copy.id).count() == 2
        assert db.query(EntityName).filter_by(entity_id=copy.id, kind='FORMER').count() == 1
    finally:
        world_service.delete_world(db, restored)


def test_http_results_and_unapproved_proposal(project):
    db, world = project
    client = TestClient(app)
    response = client.post(f'/api/worlds/{world.id}/entity-operations', json={'idempotency_key':'http-success','operations':[rename().model_dump()]})
    assert response.status_code == 200 and response.json()['status'] == 'committed'
    proposal = Op(action='rename', entity_ref='Planet X', value='Neso', decision_origin='HYPOTHETICAL')
    result = apply(project, proposal)
    assert result['status'] == 'proposed' and value(project) == 'Pallene'
    assert 'not been added' in result['message']


@pytest.mark.parametrize('scope', ['world', 'culture-a'])
def test_shared_character_names_and_reservations_remain_scoped(project, scope):
    apply(project, Op(action='reserve_name', entity_ref='Character A', entity_type='character', scope_key=scope, value='Lianne'))
    with pytest.raises(LocalModelError):
        apply(project, rename(ref='Character B', value='Lianne', scope_key=scope))
    apply(project, rename(ref='Character B', value='Lianne', scope_key='elsewhere'))


def test_reservation_is_not_assignment_and_can_be_released(project):
    apply(project, Op(action='reserve_name', entity_ref='Planet Y', entity_type='planet', value='Enceladus'))
    assert not project[0].query(EntityProperty).filter_by(entity_id=get_entity(project,'Planet Y').id).count()
    apply(project, Op(action='release_name', entity_ref='Planet Y', value='Enceladus'), rename(value='Enceladus'))
    assert value(project) == 'Enceladus'


def test_canonical_assignment_keeps_its_explicit_reservation(project):
    apply(project, Op(action='reserve_name', entity_ref='Character A', entity_type='character', value='Lianne'))
    apply(project, rename(ref='Character A', value='Lianne'))
    with pytest.raises(LocalModelError):
        apply(project, rename(ref='Character B', value='Lianne'))


def test_policy_alias_uniqueness_does_not_make_character_names_globally_unique(project):
    apply(project, rename(ref='Character A', value='Lianne'), rename(ref='Character B', value='Lianne'))
    db, world = project
    client = TestClient(app)
    response = client.put(f'/api/worlds/{world.id}/naming-policies/character', json={'canonical_unique':False,'aliases_unique':True})
    assert response.status_code == 200
    response = client.put(f'/api/worlds/{world.id}/naming-policies/character', json={'canonical_unique':True})
    assert response.status_code == 409


def test_naming_conflict_http_error_and_idempotent_failed_retry(project):
    apply(project, rename(ref='Planet Y', value='Enceladus'))
    db, world = project
    payload = {'idempotency_key':'conflicting-retry','operations':[rename(value='Enceladus').model_dump()]}
    client = TestClient(app)
    assert client.post(f'/api/worlds/{world.id}/entity-operations',json=payload).status_code == 409
    assert client.post(f'/api/worlds/{world.id}/entity-operations',json=payload).status_code == 409
    assert db.query(CanonOperationBatch).filter_by(world_id=world.id, idempotency_key='conflicting-retry').one().status == 'failed'


def test_unapproved_suggestion_and_explicit_approval_preserve_provenance(project):
    apply(project, rename())
    db, world = project
    proposal = Op(action='rename', entity_ref='Planet X', value='Neso', decision_origin='SUGGESTION')
    result = apply(project, proposal)
    stored = db.get(CanonOperationBatch, result['batch_id'])
    client = TestClient(app)
    response = client.post(f'/api/worlds/{world.id}/entity-operation-batches/{stored.id}/approve', json={
        'idempotency_key':'approved-proposal', 'operations':json.loads(stored.operations_json)})
    assert response.status_code == 200
    assert value(project) == 'Neso'
    prop = db.query(EntityProperty).filter_by(entity_id=get_entity(project).id, property_key='canonical_name').one()
    assert prop.provenance == 'AI_GENERATED_AUTHOR_APPROVED'
    repeated = client.post(f'/api/worlds/{world.id}/entity-operation-batches/{stored.id}/approve', json={
        'idempotency_key':'approved-again','operations':json.loads(stored.operations_json)})
    assert repeated.status_code == 200 and repeated.json() == response.json()


def test_stale_proposal_approval_does_not_overwrite_new_author_decision(project):
    apply(project, rename())
    result = apply(project, Op(action='rename',entity_ref='Planet X', value='Suggested',decision_origin='SUGGESTION'))
    stored = project[0].get(CanonOperationBatch,result['batch_id'])
    ops = json.loads(stored.operations_json)
    apply(project, rename(value='AuthorChoice'))
    response = TestClient(app).post(f'/api/worlds/{project[1].id}/entity-operation-batches/{stored.id}/approve',json={
        'idempotency_key':'stale-approval','operations':ops})
    assert response.status_code == 409 and value(project) == 'AuthorChoice'


def test_legacy_proposal_uses_same_registry_and_versions(project):
    from backend.schemas import ProposalActionRequest
    apply(project, rename())
    db, world = project
    proposal = bible_service.stage_proposal(db,world.id,'Proposed name','Planet X is named Neso.','AI_PROPOSED_UNVERIFIED')
    apply(project, rename(value='AuthorChoice'))
    with pytest.raises(LocalModelError):
        bible_service.resolve_proposal(db,proposal,ProposalActionRequest(action='APPROVE'))
    db.rollback()
    assert value(project) == 'AuthorChoice'


def test_markdown_only_legacy_owner_prevents_duplicate_assignment(project):
    db, world = project
    section = db.query(BibleSection).filter_by(world_id=world.id,slug='cosmology-physics').one()
    section.markdown_content = 'Planet Y is called Enceladus.'; db.commit()
    with pytest.raises(LocalModelError):
        apply(project,rename(value='Enceladus'))
    assert section.markdown_content == 'Planet Y is called Enceladus.'
    apply(project,rename(ref='Planet Y',value='NewName'))
    assert 'Planet Y is called NewName.' in db.get(BibleSection,section.id).markdown_content
    assert 'Planet Y is called Enceladus.' not in db.get(BibleSection,section.id).markdown_content
    assert 'Former name: Enceladus' in db.get(BibleSection,section.id).markdown_content


def test_manual_conflict_preserves_text_without_invalidating_other_entity(project):
    apply(project, rename(), Op(action='reserve_name',entity_ref='Planet Y',entity_type='planet',value='Enceladus'))
    db, world = project
    section = db.query(BibleSection).filter_by(world_id=world.id,slug='cosmology-physics').one()
    manual = section.markdown_content.replace('Planet X is called Pallene.', 'Planet X is called Enceladus.')
    bible_service.update_section_directly(db,section,BibleSectionUpdate(markdown_content=manual,expected_version_number=section.version_number))
    assert section.markdown_content == manual
    assert get_entity(project).canon_status == 'NEEDS_REVIEW'
    assert get_entity(project,'Planet Y').canon_status == 'ESTABLISHED_CANON'


def test_manual_uncertainty_in_one_property_keeps_other_properties(project):
    apply(project,rename(ref='Character A',value='Lianne'),Op(action='set_property',entity_ref='Character A',property_key='age',value=40))
    db, world = project
    section = db.query(BibleSection).filter_by(world_id=world.id,slug='cultures-societies').one()
    manual = section.markdown_content.replace("Character A's age is 40.", 'Character A might be older; I have not decided.')
    bible_service.update_section_directly(db,section,BibleSectionUpdate(markdown_content=manual,expected_version_number=section.version_number))
    name = db.query(EntityProperty).filter_by(entity_id=get_entity(project,'Character A').id,property_key='canonical_name').one()
    age = db.query(EntityProperty).filter_by(entity_id=name.entity_id,property_key='age').one()
    assert name.canon_status == 'ESTABLISHED_CANON' and age.canon_status == 'UNRESOLVED_POSSIBILITY'


def test_entity_id_and_label_cannot_target_different_entities(project):
    apply(project,rename(),rename(ref='Planet Y',value='Enceladus'))
    with pytest.raises(LocalModelError) as exc:
        apply(project,rename(entity_id=get_entity(project,'Planet Y').id,value='Unrequested'))
    assert exc.value.error_code == 'ENTITY_REFERENCE_MISMATCH'


def test_model_cannot_invent_scope_or_entity_type_to_bypass_names(project):
    db, world = project
    message = 'Planet X is called Pallene.'
    turn = reconcile_turn(message,TurnAnalysisSchema(primary_intent='CANON_DECLARATION',creative_permission=CreativePermissionSchema(level='FACTUAL_ONLY')))
    for updates in ({'scope_key':'made-up-scope'}, {'entity_type':'city'}):
        with pytest.raises(LocalModelError):
            entity_service.execute(db,world.id,[rename(source_quote=message,**updates)],str(uuid4()),analysis=turn,author_text=message)
    assert not db.query(Entity).filter_by(world_id=world.id).count()


def test_source_quote_does_not_borrow_a_value_from_another_clause(project):
    db, world = project
    message = 'Planet X is called Pallene. Planet Y is called Enceladus.'
    turn = reconcile_turn(message,TurnAnalysisSchema(primary_intent='CANON_DECLARATION',creative_permission=CreativePermissionSchema(level='FACTUAL_ONLY')))
    with pytest.raises(LocalModelError):
        entity_service.execute(db,world.id,[rename(value='Enceladus',source_quote='Planet X is called Pallene.')], 'wrong-clause', analysis=turn,author_text=message)


def test_known_canonical_name_resolves_to_stable_entity_on_author_rename(project):
    apply(project,rename())
    db, world = project
    original = get_entity(project).id
    message = 'Rename Pallene to Neso.'
    turn = reconcile_turn(message,TurnAnalysisSchema(primary_intent='CANON_DECLARATION',creative_permission=CreativePermissionSchema(level='FACTUAL_ONLY')))
    op = Op(action='rename',entity_ref='Pallene',value='Neso',source_quote=message,expected_version=get_entity(project).version_number)
    entity_service.execute(db,world.id,[op],'alias-rename',analysis=turn,author_text=message)
    assert get_entity(project).id == original and value(project) == 'Neso'


@pytest.mark.parametrize('claim',['I have updated your World Bible.', 'Saved.', 'Planet X is now Pallene.', 'I have recorded your changes.'])
def test_model_success_claims_are_suppressed_before_commit(claim):
    from backend.services.persistence_reporting import PendingReplyFilter, pending_reply
    filtered = PendingReplyFilter()
    pieces = [filtered.feed(claim[:8]),filtered.feed(claim[8:]),filtered.finish()]
    assert not ''.join(pieces).strip()
    assert 'awaiting validation' in pending_reply(claim)


def test_real_concurrent_sessions_only_one_stale_batch_can_win(project):
    from concurrent.futures import ThreadPoolExecutor
    apply(project,rename())
    db, world = project
    entity_id, version = get_entity(project).id, get_entity(project).version_number
    world_id = world.id
    db.rollback()
    def update(name):
        with SessionLocal() as other:
            try:
                entity_service.execute(other,world_id,[Op(action='rename',entity_id=entity_id,value=name,expected_version=version)],
                    'concurrent-' + name,manual=True)
                return 'committed'
            except LocalModelError as exc:
                return exc.error_code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update,['First','Second']))
    assert sorted(results) == ['STALE_ENTITY_VERSION','committed']
    assert value(project) in {'First','Second'}


def test_treaty_protected_dependency_validates_before_change(project):
    apply(project,Op(action='set_property',entity_ref='Country A',entity_type='country',property_key='government',value='federal'),
          Op(action='create_entity',entity_ref='Country B',entity_type='country'))
    owner = get_entity(project,'Country A')
    apply(project,Op(action='add_relationship',entity_ref='Country A',related_entity_ref='Country B',relationship_type='treaty_with',
                     value={'protected_properties':{owner.id:{'government':'federal'}}}))
    with pytest.raises(LocalModelError,match='treaty'):
        apply(project,Op(action='set_property',entity_ref='Country A',property_key='government',value='unitary'))
    assert value(project,'government','Country A') == 'federal'


def test_unknown_parent_chronology_requires_clarification(project):
    apply(project,rename(ref='Character A',value='Lianne'),rename(ref='Character B',value='Nerea'))
    with pytest.raises(LocalModelError) as exc:
        apply(project,Op(action='add_relationship',entity_ref='Character A',related_entity_ref='Character B',relationship_type='parent_of'))
    assert exc.value.error_code == 'CLARIFICATION_REQUIRED'
    assert not project[0].query(EntityRelationship).filter_by(world_id=project[1].id).count()


def test_relationship_creation_order_and_removal_are_atomic(project):
    apply(project,Op(action='add_relationship',entity_ref='Country A',entity_type='country',related_entity_ref='Country B',relationship_type='allied_with'),
          Op(action='create_entity',entity_ref='Country B',entity_type='country'))
    a,b = get_entity(project,'Country A'),get_entity(project,'Country B')
    assert project[0].query(EntityRelationship).filter_by(source_entity_id=a.id,target_entity_id=b.id,canon_status='ESTABLISHED_CANON').count()==1
    apply(project,Op(action='remove_relationship',entity_id=a.id,related_entity_id=b.id,relationship_type='allied_with'))
    assert project[0].query(EntityRelationship).filter_by(source_entity_id=a.id,target_entity_id=b.id,canon_status='SUPERSEDED').count()==1


@pytest.mark.parametrize('origin,expected_provenance,expected_status',[
    ('SUGGESTION','AI_PROPOSED_UNVERIFIED','UNRESOLVED_POSSIBILITY'),
    ('HYPOTHETICAL','AUTHOR_CREATED','HYPOTHETICAL'),
    ('RESEARCH','RESEARCH_NOTE','UNRESOLVED_POSSIBILITY')])
def test_provisional_origins_never_become_canon(project,origin,expected_provenance,expected_status):
    result = apply(project,Op(action='rename',entity_ref='Planet X',value='Unapproved',decision_origin=origin))
    assert result['status']=='proposed'
    assert result['operations'][0]['provenance']==expected_provenance
    assert result['operations'][0]['canon_status']==expected_status
    assert project[0].query(Entity).filter_by(world_id=project[1].id).count()==0


def test_structured_proposal_outside_creative_scope_is_rejected(project):
    db,world = project
    message='Give me five names for Planet X.'
    turn=reconcile_turn(message,TurnAnalysisSchema(primary_intent='CREATIVE_GENERATION',creative_permission=CreativePermissionSchema(level='BOUNDED_BRAINSTORMING')))
    with pytest.raises(LocalModelError):
        entity_service.execute(db,world.id,[Op(action='rename',entity_ref='Planet Y',value='Unapproved',decision_origin='SUGGESTION',source_quote=message)],
            'unrelated-proposal',analysis=turn,author_text=message)


def test_migration_is_repeatable_and_preserves_legacy_ids_and_text(tmp_path):
    from backend.migrations import migrate,TABLES
    from sqlalchemy.orm import Session
    path=tmp_path/'legacy.db'
    engine=create_engine('sqlite:///'+str(path))
    Base.metadata.create_all(engine,tables=[table for table in Base.metadata.sorted_tables if table.name not in TABLES])
    with Session(engine) as db:
        db.add(World(id='legacy-world',name='Old project'));db.flush()
        db.add(Fact(id='legacy-fact',world_id='legacy-world',title='Original',content='Original prose.',provenance_type='AUTHOR_CREATED'))
        db.add(BibleSection(id='legacy-section',world_id='legacy-world',title='Old text',slug='old',markdown_content='Uninterpreted author prose.'))
        db.commit()
    backup=migrate(engine)
    assert backup and __import__('pathlib').Path(backup).is_file()
    assert migrate(engine) is None
    with Session(engine) as db:
        assert db.get(Fact,'legacy-fact').content=='Original prose.'
        assert db.get(BibleSection,'legacy-section').markdown_content=='Uninterpreted author prose.'
        assert db.query(Entity).count()==0
    engine.dispose()


def test_malformed_archive_cannot_reference_another_world_or_leave_partial_import(project):
    apply(project,rename())
    db,world=project
    archive=world_service.export_project_bundle(db,world)
    archive['structured_canon']['entity_properties'][0]['entity_id']='unexported-entity'
    count=db.query(World).count()
    with pytest.raises(ValueError):
        world_service.import_project_bundle(db,archive)
    assert db.query(World).count()==count


def test_rejected_name_can_only_be_reintroduced_by_explicit_author_choice(project):
    apply(project,Op(action='reject_name',entity_ref='Planet X',entity_type='planet',value='Enceladus'))
    db,world=project
    message='You decide what Planet X should be called.'
    turn=reconcile_turn(message,TurnAnalysisSchema(primary_intent='CREATIVE_GENERATION',creative_permission=CreativePermissionSchema(level='BOUNDED_BRAINSTORMING')))
    op=rename(value='Enceladus',source_quote=message,decision_origin='DELEGATED',decision_target='Planet X',expected_version=get_entity(project).version_number)
    with pytest.raises(LocalModelError):
        entity_service.execute(db,world.id,[op],'rejected-delegation',analysis=turn,author_text=message,assistant_text='Planet X is named Enceladus.')
    apply(project,rename(value='Enceladus'))
    assert value(project)=='Enceladus'
