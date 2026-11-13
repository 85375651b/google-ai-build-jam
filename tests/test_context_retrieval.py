"""Selection, provenance, budget and focused-edit regressions over isolated data."""
from datetime import datetime, timedelta, timezone
import json
import logging

import pytest
from sqlalchemy import text

from backend.config import settings
from backend.database import SessionLocal
from backend.models import (BibleSection, CanonProposal, Conversation, Fact, FactCitation,
                            FactRelationship, FactSubject, Message, ReferenceChunk, ReferenceDocument,
                            Subject, UnresolvedQuestion)
from backend.prompts.templates import PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT
from backend.schemas import (BibleSectionEditItem, CreativePermissionSchema, PostTurnExtractionSchema,
                             TurnAnalysisSchema, WorldCreate)
from backend.services.agent_orchestrator import agent_orchestrator
from backend.services.bible_service import bible_service
from backend.services.author_intent import reconcile_turn
from backend.services.book_ingestion_service import book_ingestion_service
from backend.services.context_assembler import context_assembler
from backend.services.context_retrieval import ContextBudgetError, context_retriever, estimate_tokens
from backend.services.llm_service import LocalModelError, llm_service
from backend.services.research_service import research_service
from backend.services.world_service import world_service

BOOK = {'coverage_status': 'NOT_REQUIRED', 'passages': []}
WEB = {'status': 'SKIPPED', 'sources': []}


@pytest.fixture
def project():
    with SessionLocal() as db:
        world = world_service.create_world(db, WorldCreate(name='Selective context test'))
        conversation = db.query(Conversation).filter_by(world_id=world.id).one()
        yield db, world, conversation
        db.rollback()
        world_service.delete_world(db, world)


def analysis(query, **kwargs):
    value = TurnAnalysisSchema(primary_intent='IMPLEMENTATION_ASSISTANCE',
        creative_permission=CreativePermissionSchema(level='ANALYTICAL_ADVISORY'),
        requires_factual_retrieval=False, search_queries=[query], **kwargs)
    return reconcile_turn(query, value)


def fact(db, world, title, content, **kwargs):
    value = Fact(world_id=world.id, title=title, content=content, **kwargs)
    db.add(value); db.flush()
    return value


def select(project, query):
    db, world, conversation = project
    return context_retriever.retrieve(db, world.id, query, analysis(query), conversation)


def rendered(bundle):
    return '\n\n'.join(item.text for item in bundle.items)


def assemble(project, query, **kwargs):
    db, world, conversation = project
    return context_assembler.assemble_consultation(db, world, analysis(query), BOOK, WEB, conversation, query, **kwargs)


def test_relevant_planet_not_recent_unrelated_facts(project):
    db, world, _ = project
    relevant = fact(db, world, 'Planet X atmosphere', 'Planet X has an argon atmosphere.')
    for index in range(40):
        fact(db, world, f'Unrelated character {index}', f'Captain Zed {index} lives in an unrelated country.')
    db.commit()
    bundle = select(project, 'Tell me about Planet X.')
    assert relevant.id in {item.id for item in bundle.items}
    assert 'argon atmosphere' in rendered(bundle)
    assert 'Captain Zed' not in rendered(bundle)


def test_cross_entity_constraint_and_bounded_graph_expansion(project):
    db, world, _ = project
    target = fact(db, world, 'Planet X orbit', 'Planet X orbits Helion.')
    reservation = fact(db, world, 'Planet Y reservation', 'Enceladus is reserved for Planet Y. Planet X cannot use it.')
    unrelated = fact(db, world, 'Planet Z oceans', 'Planet Z has purple oceans.')
    db.add_all([FactRelationship(world_id=world.id, source_fact_id=target.id, target_fact_id=reservation.id,
                                relationship_type='CONSTRAINS'),
                FactRelationship(world_id=world.id, source_fact_id=reservation.id, target_fact_id=unrelated.id,
                                 relationship_type='CROSS_REFERENCES')])
    db.commit()
    bundle = select(project, 'Help me name Planet X.')
    text = rendered(bundle)
    assert 'Planet Y' in text and 'reserved' in text and 'cannot use it' in text
    assert 'Planet Z' not in text
    assert 'CONSTRAINS' in text and any('Planet Y' in name for name in bundle.related_entities)


def test_parent_entity_and_sibling_reservation_without_graph(project):
    db, world, _ = project
    fact(db, world, 'X system', 'Planet X belongs to the Helion System.')
    fact(db, world, 'Y system', 'Planet Y belongs to the Helion System. Its name Enceladus is reserved.')
    fact(db, world, 'Z system', 'Planet Z belongs to the Distant System and has neon clouds.')
    db.commit()
    text = rendered(select(project, 'Help me think of names for Planet X.'))
    assert 'Planet Y' in text and 'Enceladus' in text
    assert 'Planet Z' not in text


def test_entity_subject_properties_parent_and_alias(project):
    db, world, _ = project
    parent = Subject(world_id=world.id, name='Thalassian Culture', slug='thalassian', description='Uses open vowels.')
    db.add(parent); db.flush()
    child = Subject(world_id=world.id, name='Orelia', slug='orelia', parent_subject_id=parent.id, description='A coastal republic.')
    db.add(child); db.flush()
    value = fact(db, world, 'Local titles', 'The ruling council serves for seven years.')
    db.add(FactSubject(fact_id=value.id, subject_id=child.id)); db.commit()
    bundle = select(project, 'Discuss Orelia and its political system.')
    assert {'Orelia'} <= bundle.entities
    assert 'Thalassian Culture' in bundle.related_entities
    assert 'seven years' in rendered(bundle) and 'Uses open vowels' in rendered(bundle)


def test_old_conversation_retrieval_and_pronoun_continuity(project):
    db, world, conversation = project
    fact(db, world, 'Lianne', 'Lianne is the harbor master.')
    old = Conversation(world_id=world.id, title='Old naming decision', is_archived=True)
    db.add(old); db.flush()
    old_message = Message(conversation_id=old.id, role='USER',
        content='Lianne\'s mother chose that name to honor her grandmother.',
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    db.add(old_message)
    for index in range(35):
        other = Conversation(world_id=world.id, title=f'Unrelated {index}')
        db.add(other); db.flush()
        db.add(Message(conversation_id=other.id, role='USER', content=f'Copper coin engraving process {index}.'))
    db.add_all([Message(conversation_id=conversation.id, role='USER', content='Let us discuss Lianne.'),
                Message(conversation_id=conversation.id, role='ASSISTANT', content='We can consider Lianne\'s background.')])
    db.commit()
    bundle = select(project, 'Remember how we decided why her mother chose that name?')
    assert old_message.id in {item.id for item in bundle.items if item.kind == 'history'}
    assert 'honor her grandmother' in rendered(bundle)
    assert 'Copper coin' not in rendered(bundle)
    assert len(bundle.recent) == 2


def test_global_reservation_and_rejected_name_retrieval(project):
    db, world, _ = project
    fact(db, world, 'Name reservation', 'Enceladus is a name reserved for another planet.')
    fact(db, world, 'Rejected name', 'Reject Pallene as a name for Planet X.', canon_status='REJECTED')
    db.commit()
    text = rendered(select(project, 'Help me name Planet X.'))
    assert 'Enceladus' in text and 'Status: REJECTED' in text
    assert analysis('Planet X is not Enceladus.').creative_permission.authorization_kind == 'NONE'


@pytest.mark.parametrize('status,provenance', [
    ('ESTABLISHED_CANON', 'AUTHOR_CREATED'), ('ESTABLISHED_CANON', 'AI_GENERATED_AUTHOR_APPROVED'),
    ('HYPOTHETICAL', 'AI_PROPOSED_UNVERIFIED'), ('REJECTED', 'AUTHOR_CREATED'),
    ('SUPERSEDED', 'AUTHOR_CREATED'), ('PENDING_RETIREMENT_CONFIRMATION', 'AUTHOR_CREATED')])
def test_canon_status_and_provenance_preserved(project, status, provenance):
    db, world, _ = project
    fact(db, world, 'Planet X naming', 'Planet X has the proposed name Pallene.', canon_status=status, provenance_type=provenance)
    db.commit()
    text = rendered(select(project, 'Discuss Planet X.'))
    assert f'Status: {status}' in text and f'Provenance: {provenance}' in text
    if status != 'ESTABLISHED_CANON':
        system, _, _ = assemble(project, 'Discuss Planet X.')
        canon = system.split('<established_world_canon')[1].split('</established_world_canon>')[0]
        assert 'proposed name Pallene' not in canon


def test_proposals_questions_and_ai_history_are_not_canon(project):
    db, world, conversation = project
    db.add_all([CanonProposal(world_id=world.id, proposed_title='Planet X', proposed_content='Pallene might work.',
                             proposed_provenance='AI_PROPOSED_UNVERIFIED', status='PENDING'),
                UnresolvedQuestion(world_id=world.id, question_text='Does Planet X have rings?'),
                Message(conversation_id=conversation.id, role='ASSISTANT', content='I suggest Pallene for Planet X.',
                        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))])
    for index in range(8):
        db.add(Message(conversation_id=conversation.id, role='USER', content=f'Unrelated pottery {index}.'))
    db.commit()
    text = rendered(select(project, 'What is known about Planet X?'))
    assert 'Status: PENDING' in text and 'NOT CANON' in text
    assert 'AI SUGGESTION: not author evidence' in text and 'Does Planet X have rings?' in text


def test_relevant_section_passage_near_end_and_negation_preserved(project):
    db, world, _ = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    section.markdown_content = '\n\n'.join(['### Remote worlds'] + [f'Distant object {i} has an ordinary orbit.' for i in range(100)] +
        ['### Planet X', 'Planet X must never be called Enceladus. That name belongs to Planet Y.'])
    db.commit()
    text = rendered(select(project, 'Discuss Planet X.'))
    assert 'must never' in text and 'That name belongs to Planet Y.' in text
    assert '### Planet X' in text and 'Cosmology' in text
    assert 'Distant object' not in text


def test_summary_once_and_hypotheses_retain_status(project):
    db, _, conversation = project
    conversation.rolling_summary = 'AUTHOR DECISION: Planet X is Pallene.\n\nUNRESOLVED: Maybe Planet Y has rings.'
    db.add(Message(conversation_id=conversation.id, role='USER', content='Planet X is Pallene.'))
    db.commit()
    system, contents, diagnostic = assemble(project, 'Tell me about Planet X.')
    context = system + '\n' + '\n'.join(message['content'] for message in contents)
    assert context.count('Planet X is Pallene.') == 1
    assert 'UNRESOLVED: Maybe Planet Y has rings.' in context
    assert diagnostic['rolling_summary_included']
    assert 'Rolling Summary' not in system


def test_reference_book_relevance_and_page_citations(project):
    db, world, conversation = project
    # Real FTS/BM25 and local reference retrieval, not a mocked passage selector.
    book = book_ingestion_service.retrieve_passages(db, ['rain shadow monsoon climate'], ['geography_geology'], top_k=3)
    assert book['passages']
    system = context_assembler.assemble_consultant_system_prompt(db, world, analysis('Explain rain shadow monsoon climate.'),
        book, WEB, conversation=conversation, latest_user_message='Explain rain shadow monsoon climate.')
    for passage in book['passages']:
        assert passage['citation_id'] in system
        assert f'book_page="{passage["book_page_start"]}"' in system
        assert passage['content'] in system
    assert all('rain' in p['content'].lower() or 'climate' in p['content'].lower() or 'monsoon' in p['content'].lower() for p in book['passages'])


def test_straight_canon_update_skips_reference_and_online_research(project, monkeypatch):
    db, _, _ = project
    def unexpected(*args, **kwargs):
        pytest.fail('Canon-only operation must not retrieve factual research.')
    monkeypatch.setattr(book_ingestion_service, 'ensure_primary_book_ingested', unexpected)
    result = agent_orchestrator.retrieve_knowledge(db, analysis('Planet X is called Pallene.'))
    assert not result['book']['passages'] and result['online']['status'] == 'SKIPPED_NOT_REQUIRED'


def test_extraction_context_relevant_inputs_and_safe_section_rebase(project):
    db, world, conversation = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    section.markdown_content = 'Unrelated distant star has a blue corona.\n\nPlanet X is Enceladus.\n\nAn unrelated moon has oceans.'
    other = db.query(BibleSection).filter_by(world_id=world.id, slug='politics-governance').one()
    other.markdown_content = 'The distant republic elects an assembly.'
    fact(db, world, 'Planet Y reservation', 'Enceladus belongs to Planet Y, not Planet X.')
    user = Message(conversation_id=conversation.id, role='USER', content='Actually, Planet X is Pallene, not Enceladus.')
    assistant = Message(conversation_id=conversation.id, role='ASSISTANT', content='I will record your correction.')
    db.add_all([user, assistant]); db.commit()
    prompt, bundle, diagnostic = context_assembler.assemble_extraction(db, world, conversation, user, assistant, analysis(user.content),
        PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT, PostTurnExtractionSchema)
    assert user.content in prompt and assistant.content in prompt
    assert 'Enceladus belongs to Planet Y' in prompt and '"creative_permission"' in prompt
    assert 'The distant republic' not in prompt and 'blue corona' not in prompt
    assert 'An unrelated moon' not in prompt
    extracted = PostTurnExtractionSchema(bible_section_updates=[BibleSectionEditItem(
        section_slug=section.slug, section_title=section.title, updated_markdown='Planet X is Pallene.', source_quote=user.content)])
    context_assembler.expand_extracted_section_edits(extracted, bundle)
    assert extracted.bible_section_updates[0].updated_markdown == 'Unrelated distant star has a blue corona.\n\nPlanet X is Pallene.\n\nAn unrelated moon has oceans.'
    assert diagnostic['stage'] == 'extraction'


def test_extraction_cannot_replace_an_unseen_existing_section(project):
    bundle = select(project, 'Discuss Planet X.')
    extracted = PostTurnExtractionSchema(bible_section_updates=[BibleSectionEditItem(
        section_slug='politics-governance', section_title='Politics', updated_markdown='Changed.', source_quote='Changed.')])
    with pytest.raises(LocalModelError) as failure:
        context_assembler.expand_extracted_section_edits(extracted, bundle)
    assert failure.value.error_code == 'EXTRACTION_CONTEXT_SCOPE'


def test_budget_overflow_does_not_drop_direct_canon_or_prohibition(project, monkeypatch):
    db, world, _ = project
    fact(db, world, 'Planet X mandatory constraints', 'Planet X must not be renamed. ' + 'Required established property. ' * 700)
    db.commit()
    monkeypatch.setenv('CONSULTATION_CONTEXT_TOKENS', '3500')
    with pytest.raises(ContextBudgetError) as error:
        assemble(project, 'Tell me about Planet X.')
    assert error.value.error_code == 'CONTEXT_BUDGET_EXCEEDED'
    assert 'not discarded' in error.value.message


def test_diagnostics_contain_counts_not_private_text(project, caplog):
    db, world, _ = project
    fact(db, world, 'Planet X', 'Planet X secret manuscript phrase.')
    db.commit()
    with caplog.at_level(logging.INFO, logger='backend.services.context_retrieval'):
        _, _, diagnostic = assemble(project, 'Discuss Planet X.')
    assert diagnostic['facts'] == 1 and diagnostic['entities'] > 0
    assert diagnostic['estimated_input_tokens'] <= diagnostic['budget']
    assert 'secret manuscript phrase' not in caplog.text
    assert 'component_tokens' in caplog.text and 'selected_record_tokens' in caplog.text


@pytest.mark.parametrize('scenario', ['short naming correction', 'factual research', 'character development',
                                   'political worldbuilding', 'cross-entity consistency', 'historical conversation'])
def test_representative_requests_exclude_bulk_and_reduce_context(project, scenario):
    from context_scenarios import SCENARIOS, seed_context_scenarios
    db, world, conversation = project
    seed_context_scenarios(db, world, conversation)
    query = SCENARIOS[scenario]
    system, contents, diagnostic = assemble(project, query)
    # The previous dynamic context alone: every section's first 1,500 chars,
    # 25 newest facts, 12 messages and duplicate summary. This excludes its
    # immutable instructions, making the reduction assertion conservative.
    legacy = '\n'.join(s.markdown_content[:1500] for s in db.query(BibleSection).filter_by(world_id=world.id))
    legacy += '\n'.join(f.content for f in db.query(Fact).filter_by(world_id=world.id).order_by(Fact.updated_at.desc()).limit(25))
    legacy += '\n'.join(m.content for m in db.query(Message).filter_by(conversation_id=conversation.id).order_by(Message.created_at.desc()).limit(12))
    legacy += conversation.rolling_summary * 2
    assert diagnostic['estimated_input_tokens'] < estimate_tokens(legacy) * 0.65
    assert 'Ceramic glazes' not in system
    if scenario in {'short naming correction', 'cross-entity consistency'}:
        assert 'reserved for Planet Y' in system
    if scenario == 'historical conversation':
        assert 'honor her grandmother' in system
    assert diagnostic['estimated_input_tokens'] <= diagnostic['budget']


def test_supplementary_research_relevance_and_citation_integrity():
    relevant = {'citation_id': 'Web-2', 'title': 'Monsoon climate', 'url': 'https://example.org/climate',
                'excerpt': 'Monsoon rainfall varies seasonally.\n\nHowever, mountains change its distribution.'}
    irrelevant = {'citation_id': 'Web-3', 'title': 'Pottery kilns', 'url': 'https://example.org/pottery',
                  'excerpt': 'Ceramic glazes require heat.'}
    book, web = context_retriever.select_research(BOOK, {'sources': [relevant, irrelevant, relevant]}, ['monsoon climate'])
    assert web['sources'] == [relevant]
    assert 'However' in web['sources'][0]['excerpt'] and web['sources'][0]['citation_id'] == 'Web-2'


def test_extraction_budget_fails_before_inference_and_keeps_database(project, monkeypatch):
    db, world, conversation = project
    fact(db, world, 'Planet X', 'Planet X is subject to an immutable constraint. ' * 700)
    user = Message(conversation_id=conversation.id, role='USER', content='Planet X is called Pallene.')
    assistant = Message(conversation_id=conversation.id, role='ASSISTANT', content='I will record that.')
    db.add_all([user, assistant]); db.commit()
    monkeypatch.setenv('EXTRACTION_CONTEXT_TOKENS', '4000')
    monkeypatch.setattr(llm_service, 'generate_structured', lambda **kwargs: pytest.fail('Unsafe context must not reach inference.'))
    with pytest.raises(ContextBudgetError):
        agent_orchestrator.prepare_post_turn_extraction(db, world, conversation, user, assistant, analysis(user.content))
    assert db.query(Fact).filter_by(world_id=world.id).count() == 1


def test_lowercase_political_entity_is_essential(project):
    db, world, _ = project
    fact(db, world, 'Treaty', 'The northern kingdom and the coastal republic signed a mutual defense treaty.')
    db.commit()
    bundle = select(project, 'Discuss the northern kingdom and coastal republic.')
    assert any(item.kind == 'fact' and item.essential for item in bundle.items)


def test_markdown_heading_without_blank_line_and_following_qualification(project):
    db, world, _ = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    section.markdown_content = '# Astronomy\n## Planet X\nPlanet X could be called Pallene.\n\nHowever, this is hypothetical and must not become canon.\n\nUnrelated ceramic ledger.'
    db.commit()
    text = rendered(select(project, 'Discuss Planet X.'))
    assert 'could be called Pallene' in text
    assert 'hypothetical and must not become canon' in text
    assert '# Astronomy' in text and '## Planet X' in text
    assert 'Unrelated ceramic' not in text


def test_section_only_relationship_expansion(project):
    db, world, _ = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cultures-societies').one()
    section.markdown_content = "Lianne's mother is Nerea.\n\nNerea raised her daughter beside the harbor.\n\nZed collects pottery."
    db.commit()
    bundle = select(project, "Discuss Lianne's family relationships.")
    assert 'Nerea' in bundle.related_entities
    assert 'raised her daughter' in rendered(bundle)
    assert 'Zed collects' not in rendered(bundle)


def test_confirmed_alias_retrieves_entity_properties_without_authorizing_invention(project):
    db, world, _ = project
    fact(db, world, 'Planet X name', 'Planet X is called Pallene.')
    fact(db, world, 'Planet X atmosphere', 'Planet X has an argon atmosphere.')
    db.commit()
    query = 'Tell me about Pallene.'
    assert 'argon atmosphere' in rendered(select(project, query))
    assert analysis(query).creative_permission.authorization_kind == 'NONE'


def test_case_variants_do_not_erase_entity_references(project):
    db, world, _ = project
    fact(db, world, 'Northern Kingdom', 'The northern kingdom signed a treaty.')
    db.commit()
    bundle = select(project, 'Discuss the Northern Kingdom.')
    assert any(name.casefold() == 'northern kingdom' for name in bundle.entities)


def test_historical_correction_has_date_and_is_not_silently_budget_dropped(project):
    db, world, conversation = project
    fact(db, world, 'Planet X', 'Planet X has an argon atmosphere.')
    db.add(Message(conversation_id=conversation.id, role='USER',
        content='Actually, Planet X must not be named Enceladus.', created_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
    for index in range(10):
        db.add(Message(conversation_id=conversation.id, role='USER', content=f'Pottery ledger {index}.'))
    db.commit()
    bundle = select(project, 'Discuss Planet X.')
    historical = [item for item in bundle.items if item.kind == 'history']
    assert len(historical) == 1 and historical[0].essential
    assert 'date=2020-01-01' in historical[0].text and 'must not' in historical[0].text


def test_missing_recalled_discussion_is_explicit(project):
    bundle = select(project, 'Remember the naming decision from earlier?')
    assert any(item.kind == 'notice' and 'ask the author' in item.text for item in bundle.items)


def test_fact_citation_and_verified_metadata_keep_exact_source(project):
    db, world, _ = project
    source = db.query(ReferenceChunk).filter(ReferenceChunk.book_page_start.is_not(None)).first()
    value = fact(db, world, 'Planet X climate', 'Planet X has seasonal winds.')
    db.add(FactCitation(fact_id=value.id, citation_source_type='REFERENCE_BOOK',
                       reference_chunk_id=source.id, citation_note='Author approved climatic premise.'))
    db.commit()
    assert f'Book-p{source.book_page_start}' in rendered(select(project, 'Discuss Planet X.'))
    passages = [{'citation_id': 'Book-p1', 'citation_label': 'Page 1', 'chapter_title': 'Climate',
                 'section_title': 'Winds', 'book_page_start': 1, 'pdf_page_start': 3, 'content': 'Winds.'},
                {'citation_id': 'Book-p10', 'citation_label': 'Page 10', 'chapter_title': 'Climate',
                 'section_title': 'Monsoons', 'book_page_start': 10, 'pdf_page_start': 12, 'content': 'Monsoons.'}]
    verified = agent_orchestrator.sanitize_response_and_verify_citations('See [Book-p10].', {'passages': passages}, WEB)
    assert [citation['citation_id'] for citation in verified['citations']] == ['Book-p10']
    assert not agent_orchestrator.sanitize_response_and_verify_citations('Book knowledge [Book-p999].', {'passages': passages}, WEB)['citations']


def test_focused_update_survives_full_request_persistence(project, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.schemas import BibleSectionUpdate
    db, world, conversation = project
    section = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    original = 'A distant star has a blue corona.\n\nPlanet X is called Enceladus.\n\nA distant moon has oceans.'
    bible_service.update_section_directly(db, section, BibleSectionUpdate(markdown_content=original))
    query = 'Actually, Planet X is called Pallene.'
    captured = {}
    def structured(**kwargs):
        if kwargs['response_schema'] is TurnAnalysisSchema:
            return analysis(query)
        captured['extraction'] = kwargs['prompt']
        return PostTurnExtractionSchema(bible_section_updates=[BibleSectionEditItem(
            section_slug=section.slug, section_title=section.title,
            updated_markdown='Planet X is called Pallene.', source_quote=query)])
    monkeypatch.setattr(llm_service, 'generate_structured', structured)
    monkeypatch.setattr(llm_service, 'generate_text', lambda **kwargs: 'I will record your correction.')
    response = TestClient(app).post(f'/api/conversations/{conversation.id}/messages', json={'content': query})
    assert response.status_code == 200 and response.json()['type'] == 'complete'
    assert 'blue corona' not in captured['extraction'] and 'moon has oceans' not in captured['extraction']
    db.rollback(); db.refresh(section)
    assert section.markdown_content == original.replace('called Enceladus', 'called Pallene')


def test_category_mention_does_not_load_all_subject_facts(project):
    db, world, _ = project
    subject = db.query(Subject).filter_by(world_id=world.id, slug='geography-geology').one()
    unrelated = fact(db, world, 'Geography and Geology: Unrelated mountain', 'A distant mountain contains cobalt ore.')
    db.add(FactSubject(fact_id=unrelated.id, subject_id=subject.id)); db.commit()
    bundle = select(project, 'My world has three islands. Record this in the Geography and Geology section.')
    assert subject.name not in bundle.entities
    assert unrelated.id not in {item.id for item in bundle.items}


def test_extraction_evidence_quotes_are_explicit_without_relaxing_gate(project):
    from backend.services.author_intent import author_evidence
    db, world, conversation = project
    user = Message(conversation_id=conversation.id, role='USER', content='My world has exactly three inhabited islands.')
    assistant = Message(conversation_id=conversation.id, role='ASSISTANT', content='I will record that.')
    db.add_all([user, assistant]); db.commit()
    turn = analysis(user.content)
    prompt, _, _ = context_assembler.assemble_extraction(db, world, conversation, user, assistant, turn,
        PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT, PostTurnExtractionSchema)
    assert json.dumps([user.content]) in prompt and 'mandatory_evidence_fields' in prompt
    assert author_evidence(user.content, turn)
    assert not author_evidence('', turn)
    assert not author_evidence(user.content.rstrip('.'), turn)
    response_schema = context_assembler.extraction_response_schema(PostTurnExtractionSchema, turn)
    inference = response_schema.model_json_schema()
    assert inference['$defs']['TurnEvidenceQuote']['enum'] == [user.content]
    assert 'source_quote' in inference['$defs']['BibleSectionEditItem']['required']
    # Existing backend models/defaults/permission gates remain unchanged.
    assert PostTurnExtractionSchema.model_json_schema()['$defs']['BibleSectionEditItem']['properties']['source_quote']['default'] == ''


def test_noisy_research_flag_does_not_override_straight_canon_management(project, monkeypatch):
    db, _, _ = project
    query = 'My world has three inhabited islands. Record this in the Geography and Geology section.'
    turn = analysis(query)
    turn.requires_factual_retrieval = True
    snapshot = turn.model_dump()
    monkeypatch.setattr(book_ingestion_service, 'ensure_primary_book_ingested', lambda *args: pytest.fail('No factual support requested.'))
    result = agent_orchestrator.retrieve_knowledge(db, turn, user_message=query)
    assert result['book']['coverage_status'] == 'NOT_REQUIRED'
    assert turn.model_dump() == snapshot


def test_mixed_canon_research_keeps_reference_selection(project, monkeypatch):
    db, _, _ = project
    query = 'Actually, Planet X is Pallene. Can you research the historical origin of that name?'
    turn = analysis(query)
    turn.requires_factual_retrieval = False
    snapshot = turn.model_dump()
    calls = []
    monkeypatch.setattr(book_ingestion_service, 'ensure_primary_book_ingested', lambda *args: calls.append('book'))
    monkeypatch.setattr(book_ingestion_service, 'retrieve_passages', lambda **kwargs: {'coverage_status': 'HIGH', 'passages': []})
    agent_orchestrator.retrieve_knowledge(db, turn, user_message=query)
    assert calls == ['book'] and turn.model_dump() == snapshot


def test_naming_collision_index_preserves_existing_names_without_unrelated_lore(project):
    db, world, _ = project
    fact(db, world, 'Planet Z', 'Planet Z is called Pallene. Its oceans are purple.')
    fact(db, world, 'Hypothetical name', 'Planet Q is called Nerea.', canon_status='HYPOTHETICAL')
    db.commit()
    bundle = select(project, 'Give me five names for Planet X.')
    registry = bundle.render({'naming_registry'})
    assert 'Pallene' in registry and 'Nerea' not in registry
    assert 'Planet Z' not in rendered(bundle) and 'purple' not in rendered(bundle)


def test_explicit_mixed_research_supplements_low_book_coverage(project, monkeypatch):
    db, _, _ = project
    query = 'Planet X is called Pallene. Can you research the historical origin of that name?'
    turn = analysis(query)
    turn.requires_factual_retrieval = False
    monkeypatch.setattr(book_ingestion_service, 'ensure_primary_book_ingested', lambda *args: None)
    monkeypatch.setattr(book_ingestion_service, 'retrieve_passages', lambda **kwargs: {'coverage_status': 'LOW', 'passages': []})
    calls = []
    def online(**kwargs):
        calls.append(kwargs['queries'])
        return {'status': 'UNCONFIGURED', 'sources': []}
    monkeypatch.setattr(research_service, 'search_online', online)
    result = agent_orchestrator.retrieve_knowledge(db, turn, user_message=query)
    assert calls and result['online']['status'] == 'UNCONFIGURED'
    assert not turn.requires_factual_retrieval


def test_budget_does_not_silently_remove_a_recent_author_constraint(project, monkeypatch):
    db, world, conversation = project
    fact(db, world, 'Planet X', 'Planet X has an argon atmosphere.')
    db.add(Message(conversation_id=conversation.id, role='USER',
        content='Planet X must never be called Enceladus. ' + 'Required qualification. ' * 700))
    db.add_all([Message(conversation_id=conversation.id, role='ASSISTANT', content='Understood.'),
                Message(conversation_id=conversation.id, role='USER', content='Let us discuss Planet X.'),
                Message(conversation_id=conversation.id, role='ASSISTANT', content='Its atmosphere is argon.')])
    db.commit()
    monkeypatch.setenv('CONSULTATION_CONTEXT_TOKENS', '3500')
    with pytest.raises(ContextBudgetError):
        assemble(project, 'Tell me about Planet X.')


def test_extraction_omits_redundant_raw_classification_transcript(project):
    db, world, conversation = project
    user = Message(conversation_id=conversation.id, role='USER', content='Planet X is called Pallene.')
    assistant = Message(conversation_id=conversation.id, role='ASSISTANT', content='I will record your decision.')
    db.add_all([user, assistant]); db.commit()
    turn = analysis(user.content)
    turn.raw_user_phrasing = 'UNNEEDED_CLASSIFICATION_ECHO ' * 300
    prompt, _, _ = context_assembler.assemble_extraction(db, world, conversation, user, assistant, turn,
        PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT, PostTurnExtractionSchema)
    assert user.content in prompt and assistant.content in prompt
    assert '"creative_permission"' in prompt and 'Validated Author Decisions' in prompt
    assert 'UNNEEDED_CLASSIFICATION_ECHO' not in prompt
