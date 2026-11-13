"""Author-controlled structured canon; no model, network, or internal commits in apply()."""
import hashlib
import json
import re
import math
import unicodedata
from contextvars import ContextVar
from uuid import uuid4
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from backend.models import (Entity, EntityProperty, EntityName, EntityRelationship, EntityConstraint,
    EntityRevision, CanonOperationBatch, CanonSyncIssue, NamingPolicy, Fact, FactRevision,
    BibleSection, BibleSectionFact, BibleSectionRevision, Subject, World)
from backend.schemas import EntityOperation, CandidateProposalItem
from backend.services.author_intent import author_evidence, clear_author_decisions, delegated_evidence, suggestion_in_scope, naming_fact_supported
from backend.services.llm_service import LocalModelError

TYPES = {'planet', 'star', 'moon', 'country', 'city', 'character', 'family', 'organization',
         'religion', 'language', 'event', 'species', 'institution', 'cultural_group'}
ACTIVE = 'ESTABLISHED_CANON'
projection_enabled = ContextVar('entity_projection_enabled', default=True)
RELATIONS = {'parent_of', 'participates_in', 'treaty_with', 'allied_with', 'located_in', 'member_of', 'depends_on'}
TYPE_SECTIONS = {'planet': 'cosmology-physics', 'star': 'cosmology-physics', 'moon': 'cosmology-physics',
    'country': 'politics-governance', 'city': 'geography-geology', 'event': 'history-chronology',
    'language': 'languages-linguistics', 'religion': 'religion-mythology', 'species': 'biology-evolution'}


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def key(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def fail(message, code='CANON_CONFLICT'):
    raise LocalModelError(message, 409, code)


def mentioned(label, text):
    return bool(label and re.search(r'(?<!\w)' + re.escape(label) + r'(?!\w)', text, re.I))


class EntityService:
    def resolve(self, db, world_id, entity_id=None, ref='', required=True):
        if entity_id:
            entity = db.get(Entity, entity_id)
            if entity is None or entity.world_id != world_id:
                fail('Entity does not belong to this world.', 'ENTITY_NOT_FOUND')
            if ref and key(ref) != entity.reference_key:
                anchor = db.query(Entity).filter_by(world_id=world_id, reference_key=key(ref)).first()
                alias = db.query(EntityName).filter_by(entity_id=entity.id, normalized_name=key(ref)).first()
                if (anchor and anchor.id != entity.id) or alias is None:
                    fail('The entity ID and reference label identify different targets.', 'ENTITY_REFERENCE_MISMATCH')
            return entity
        entities = db.query(Entity).filter(Entity.world_id == world_id, Entity.reference_key == key(ref)).all()
        if not entities:
            ids = db.query(EntityName.entity_id).filter(EntityName.world_id == world_id,
                EntityName.normalized_name == key(ref), EntityName.canon_status == ACTIVE,
                EntityName.kind.in_(['CANONICAL', 'ALIAS'])).all()
            entities = db.query(Entity).filter(Entity.id.in_({row[0] for row in ids})).all()
        if len(entities) > 1:
            fail(f'More than one entity matches {ref!r}. Which one did you mean?', 'AMBIGUOUS_ENTITY')
        if not entities:
            if required:
                fail(f'Identify the entity meant by {ref!r} before changing canon.', 'ENTITY_NOT_FOUND')
            return None
        return entities[0]

    def snapshot(self, db, entity):
        def records(model, **filters):
            return [{c.name: getattr(row, c.name) for c in model.__table__.columns}
                    for row in db.query(model).filter_by(**filters).order_by(model.id)]
        return {'id': entity.id, 'reference_label': entity.reference_label, 'entity_type': entity.entity_type,
            'scope_key': entity.scope_key, 'version_number': entity.version_number,
            'names_projected': projection_enabled.get(),
            'canon_status':entity.canon_status, 'provenance':entity.provenance,
            'properties': records(EntityProperty, entity_id=entity.id), 'names': records(EntityName, entity_id=entity.id),
            'constraints': records(EntityConstraint, entity_id=entity.id),
            'relationships': [{c.name: getattr(row, c.name) for c in EntityRelationship.__table__.columns}
                for row in db.query(EntityRelationship).filter(or_(EntityRelationship.source_entity_id == entity.id,
                    EntityRelationship.target_entity_id == entity.id)).order_by(EntityRelationship.id)]}

    def context(self, db, world_id, query, naming=False):
        entities = db.query(Entity).filter_by(world_id=world_id).all()
        direct = {e.id for e in entities if mentioned(e.reference_label, query)}
        for name in db.query(EntityName).filter_by(world_id=world_id):
            if mentioned(name.name, query):
                direct.add(name.entity_id)
        selected = set(direct)
        for rel in db.query(EntityRelationship).filter(EntityRelationship.world_id == world_id,
            EntityRelationship.canon_status == ACTIVE, or_(EntityRelationship.source_entity_id.in_(direct),
                                                          EntityRelationship.target_entity_id.in_(direct))):
            selected.update((rel.source_entity_id, rel.target_entity_id))
        if naming:
            for entity in entities:
                if entity.id in direct:
                    for owner in db.query(EntityName).join(Entity, Entity.id == EntityName.entity_id).filter(
                        EntityName.world_id == world_id, Entity.entity_type == entity.entity_type,
                        EntityName.scope_key == entity.scope_key, EntityName.kind.in_(['RESERVED', 'REJECTED']),
                        EntityName.canon_status == ACTIVE):
                        selected.add(owner.entity_id)
        result = []
        for entity in entities:
            if entity.id not in selected:
                continue
            state = self.snapshot(db, entity)
            for field in ('properties', 'names', 'constraints', 'relationships'):
                state[field] = [{key: value for key, value in row.items() if key in {
                    'id', 'entity_id', 'property_key', 'value_json', 'name', 'kind', 'scope_key', 'constraint_type',
                    'source_entity_id', 'target_entity_id', 'relationship_type', 'details_json',
                    'provenance', 'canon_status', 'version_number'}} for row in state[field]]
            result.append((entity.id, dump(state), entity.id in direct))
        return result

    def _create(self, db, world_id, op, provenance, source_id):
        entity_type = (op.entity_type or op.entity_ref.split(' ', 1)[0]).casefold()
        prefix_type = op.entity_ref.split(' ', 1)[0].casefold()
        if prefix_type in TYPES and entity_type != prefix_type:
            fail('The proposed entity type contradicts the explicit author reference.', 'ENTITY_TYPE_MISMATCH')
        if entity_type not in TYPES or not op.entity_ref.strip():
            fail('Specify an entity type and an unambiguous reference label.', 'CLARIFICATION_REQUIRED')
        existing = self.resolve(db, world_id, ref=op.entity_ref, required=False)
        if existing:
            if existing.entity_type != entity_type:
                fail('That reference already identifies a different entity type.')
            return existing
        # Reuse a specifically identified non-category Subject ID when available.
        from backend.prompts.domain_cartridges import DEFAULT_WORLD_SUBJECTS
        categories = {item['slug'] for item in DEFAULT_WORLD_SUBJECTS}
        subjects = db.query(Subject).filter(Subject.world_id == world_id, Subject.name == op.entity_ref).all()
        legacy = subjects[0] if len(subjects) == 1 and subjects[0].slug not in categories else None
        entity = Entity(id=legacy.id if legacy else str(uuid4()), world_id=world_id,
            reference_label=op.entity_ref.strip(), reference_key=key(op.entity_ref), entity_type=entity_type,
            scope_key=key(op.scope_key or 'world'), legacy_subject_id=legacy.id if legacy else None,
            provenance=provenance, source_quote=op.source_quote, source_message_id=source_id, version_number=1)
        db.add(entity); db.flush()
        self._adopt_legacy_name(db, entity)
        return entity

    def _adopt_legacy_name(self, db, entity):
        # Narrow, explicit typed identity only; never a capitalized-phrase migration.
        pattern = re.compile(rf"^{re.escape(entity.reference_label)}(?:'s name is| is (?:called |named )?)([\w'’-]+(?: [\w'’-]+)*)[.]?$", re.I)
        matches = []
        for fact in db.query(Fact).filter(Fact.world_id == entity.world_id, Fact.canon_status == ACTIVE,
                                         Fact.content.ilike('%' + entity.reference_label + '%')):
            match = pattern.fullmatch(fact.content.strip().rstrip('.'))
            if match and (match[1][0].isupper() or re.search(r'\b(?:called|named|name is)\b', fact.content)) and not re.search(r'\b(?:not|and|has|because|closest|contains|orbits)\b', match[1], re.I):
                matches.append((fact, match[1]))
        if not matches:
            for section in db.query(BibleSection).filter(BibleSection.world_id == entity.world_id,
                    BibleSection.markdown_content.ilike('%' + entity.reference_label + '%')):
                for block in section.markdown_content.split('\n\n'):
                    statement = block.strip().lstrip('-* ')
                    match = pattern.fullmatch(statement.rstrip('.'))
                    if match and (match[1][0].isupper() or re.search(r'\b(?:called|named|name is)\b', statement)):
                        fact = Fact(world_id=entity.world_id, title=entity.reference_label + ': canonical_name', content=statement,
                            provenance_type=section.dominant_provenance, canon_status=ACTIVE, version_number=1)
                        db.add(fact); db.flush()
                        db.add(BibleSectionFact(section_id=section.id, fact_id=fact.id))
                        matches.append((fact, match[1]))
        if len({key(name) for _, name in matches}) > 1:
            fail('Existing name statements disagree. Confirm the existing entity/name mapping first.', 'CLARIFICATION_REQUIRED')
        if matches:
            fact, name = matches[0]
            db.add(EntityProperty(entity_id=entity.id, property_key='canonical_name', value_json=dump(name),
                fact_id=fact.id, provenance=fact.provenance_type, canon_status=ACTIVE, source_message_id=fact.source_message_id))
            db.add(EntityName(world_id=entity.world_id, entity_id=entity.id, name=name, normalized_name=key(name),
                kind='CANONICAL', scope_key=entity.scope_key, provenance=fact.provenance_type, canon_status=ACTIVE))
            db.flush()

    def _authorize(self, op, entity, analysis, author_text, assistant_text, manual, reference_context=''):
        if manual:
            if op.decision_origin == 'RESEARCH':
                return 'RESEARCH_NOTE'
            if op.decision_origin == 'SUGGESTION':
                return 'AI_PROPOSED_UNVERIFIED'
            return 'AI_GENERATED_AUTHOR_APPROVED' if manual == 'APPROVED' else 'AUTHOR_CREATED'
        if analysis is None:
            fail('Author evidence is required.', 'UNAUTHORIZED_OPERATION')
        if op.decision_origin in {'SUGGESTION', 'HYPOTHETICAL', 'RESEARCH'} or op.action == 'propose':
            if op.decision_origin == 'RESEARCH':
                return 'RESEARCH_NOTE'
            label = entity.reference_label if entity else op.entity_ref
            content = f'{label} is named {op.value}.' if op.action in {'rename', 'add_alias', 'reserve_name'} or op.property_key == 'canonical_name' else f'{label}: {op.property_key} = {op.value}'
            candidate = CandidateProposalItem(proposed_title=label, proposed_content=content,
                source_quote=op.source_quote, decision_origin='SUGGESTION')
            if op.decision_origin == 'HYPOTHETICAL' and op.source_quote in analysis.uncertain_statements and op.source_quote in author_text and mentioned(label, op.source_quote):
                return 'AUTHOR_CREATED'
            scope = ' '.join(analysis.creative_permission.allowed_targets) + ' ' + analysis.creative_permission.resolved_target
            if (mentioned(label, scope) or mentioned(op.entity_ref, scope)) and suggestion_in_scope(candidate, analysis):
                return 'AI_PROPOSED_UNVERIFIED'
            fail('This suggestion is outside the authorized task.', 'UNAUTHORIZED_OPERATION')
        label = entity.reference_label if entity else op.entity_ref
        if op.decision_origin == 'DELEGATED':
            candidate = CandidateProposalItem(proposed_title=label,
                proposed_content=f'{label} is named {op.value}.', source_quote=op.source_quote,
                decision_target=op.decision_target, decision_origin='DELEGATED')
            if op.action not in {'rename', 'set_property'} or (op.action == 'set_property' and op.property_key != 'canonical_name') or not delegated_evidence(candidate, analysis, assistant_text, reference_context):
                fail('The operation exceeds the explicit delegation.', 'UNAUTHORIZED_OPERATION')
            return 'AI_GENERATED_AUTHOR_APPROVED'
        if not author_evidence(op.source_quote, analysis):
            fail('The operation lacks an exact validated author decision.', 'UNAUTHORIZED_OPERATION')
        # Supported natural language is independently compiled below. A correct
        # quote alone cannot authorize a different target/property/value.
        compiled = self.compile_author_operations(analysis, author_text)
        if op.action == 'create_entity' and any(key(item.entity_ref) == key(label) and item.source_quote == op.source_quote for item in compiled):
            return 'AUTHOR_CREATED'
        signatures = {(item.action, key(item.entity_ref), item.property_key, dump(item.value), key(item.related_entity_ref), item.source_quote) for item in compiled}
        signature = (op.action, key(label), op.property_key, dump(op.value), key(op.related_entity_ref), op.source_quote)
        if signature not in signatures:
            quoted_target = op.entity_ref or label
            candidate = CandidateProposalItem(proposed_title=label, proposed_content=f'{quoted_target} is named {op.value}.', source_quote=op.source_quote)
            literal_evidence = op.source_quote + (' ' + reference_context if re.search(r"\b(?:let's use|let us use|go with|I choose|I pick)\b", op.source_quote, re.I) else '')
            if re.search(r"\b(?:let's use|let us use|go with)\b", op.source_quote, re.I):
                preceding = author_text.split(op.source_quote, 1)[0]
                preferences = re.findall(r"\bI (?:like|prefer|choose|pick) ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*)*)", preceding)
                if preferences:
                    literal_evidence += ' ' + preferences[-1]
            literal_value = re.search(r'(?<!\w)' + re.escape(str(op.value)) + r'(?!\w)', literal_evidence)
            target_pattern = re.escape(quoted_target)
            naming_act = re.search(rf"{target_pattern}(?:'s name is| is (?:called|named)| to be called| should be called)\b|(?:change|rename) {target_pattern}(?:'s name)? to\b|(?:let's use|let us use|go with|I choose|I pick)\b", op.source_quote, re.I)
            if op.action == 'rename' and naming_act and literal_value and mentioned(quoted_target, op.source_quote) and not re.search(r"\b(?:not|never|cannot|can't|don't|do not)\b", op.source_quote, re.I) and naming_fact_supported(candidate, author_text, reference_context):
                return 'AUTHOR_CREATED'
            if op.action == 'set_property' and re.search(rf"{target_pattern}'s {re.escape(op.property_key)} is {re.escape(str(op.value))}[.!]?$", op.source_quote):
                return 'AUTHOR_CREATED'
            fail('The structured interpretation needs author confirmation before applying it.', 'CLARIFICATION_REQUIRED')
        return 'AUTHOR_CREATED'

    def compile_author_operations(self, analysis, author_text):
        from backend.services.canon_language import compile_decisions
        return compile_decisions(clear_author_decisions(analysis), author_text)

    def apply(self, db, world_id, operations, idempotency_key, **kwargs):
        db.info['entity_projection_slots'] = {}
        from backend.services.generation_control import current_generation
        control = current_generation.get()
        if control is not None:
            control.pending_canon_batch = {'world_id':world_id, 'idempotency_key':idempotency_key,
                'operations_json':dump([op.model_dump() for op in operations])}
        return self._apply(db, world_id, operations, idempotency_key, **kwargs)

    def _apply(self, db, world_id, operations, idempotency_key, *, analysis=None, author_text='',
              assistant_text='', source_message_id=None, expected_versions=None, manual=False, reference_context=''):
        """Validate/apply inside caller's real write transaction. Flushes, never commits."""
        payload = [op.model_dump() for op in operations]
        fingerprint = hashlib.sha256(dump(payload).encode()).hexdigest()
        previous = db.query(CanonOperationBatch).filter_by(world_id=world_id, idempotency_key=idempotency_key).first()
        if previous:
            if previous.fingerprint != fingerprint:
                fail('The operation batch ID was reused with different changes.', 'BATCH_ID_CONFLICT')
            result = json.loads(previous.result_json)
            if previous.status in {'failed', 'clarification_required'}:
                fail(result['message'], 'PREVIOUS_BATCH_FAILED')
            return result
        if db.get(World, world_id) is None:
            fail('World not found.', 'WORLD_NOT_FOUND')
        batch = CanonOperationBatch(world_id=world_id, idempotency_key=idempotency_key, fingerprint=fingerprint,
                                    status='accepted', operations_json=dump(payload))
        db.add(batch); db.flush()
        before, affected, results = {}, {}, []
        initial_versions = {}
        snapshot_supplied = expected_versions is not None
        expected_versions = dict(expected_versions or {})
        if snapshot_supplied:
            expected_versions['_strict_snapshot'] = True
        ordered = sorted(enumerate(operations), key=lambda item: item[1].action in {'add_relationship', 'remove_relationship'})
        for index, op in ordered:
            entity = self.resolve(db, world_id, op.entity_id, op.entity_ref, required=False)
            if not manual and op.scope_key is not None and key(op.scope_key) != (entity.scope_key if entity else 'world'):
                fail('A new naming scope was not authorized.', 'UNAUTHORIZED_OPERATION')
            if entity and op.entity_type and op.entity_type.casefold() != entity.entity_type:
                fail('Entity type does not match the existing identity.', 'ENTITY_TYPE_MISMATCH')
            if op.related_entity_id and not op.related_entity_ref:
                related = self.resolve(db, world_id, op.related_entity_id)
                op = op.model_copy(update={'related_entity_ref': related.reference_label})
            provenance = self._authorize(op, entity, analysis, author_text, assistant_text, manual, reference_context)
            if op.action == 'propose' or op.decision_origin in {'SUGGESTION', 'HYPOTHETICAL', 'RESEARCH'}:
                if entity:
                    op = op.model_copy(update={'entity_id': entity.id, 'expected_version': entity.version_number})
                results.append({'index': index, 'state': 'proposed', 'operation': op.model_dump(), 'provenance': provenance,
                    'canon_status': 'HYPOTHETICAL' if op.decision_origin == 'HYPOTHETICAL' else 'UNRESOLVED_POSSIBILITY'})
                continue
            if entity is None:
                if not manual and not re.match(r'^(?:Planet|Star|Moon|Country|City|Character|Family|Organization|Religion|Language|Event|Species|Institution)\s+\S+', op.entity_ref, re.I):
                    fail('Confirm the identity and type of this new entity.', 'CLARIFICATION_REQUIRED')
                entity = self._create(db, world_id, op, provenance, source_message_id)
                before[entity.id] = None
                initial_versions[entity.id] = entity.version_number
            if entity.id not in initial_versions:
                version = expected_versions.get(entity.id) if snapshot_supplied else op.expected_version
                if version is None or version != entity.version_number:
                    fail(f'{entity.reference_label} changed or its expected version was not supplied. Reload before retrying.', 'STALE_ENTITY_VERSION')
                initial_versions[entity.id] = entity.version_number
                before[entity.id] = self.snapshot(db, entity)
            affected[entity.id] = entity
            if op.scope_key is not None and key(op.scope_key) != entity.scope_key:
                fail('Changing naming scope requires a separate confirmed entity migration.', 'CLARIFICATION_REQUIRED')
            # Release old claim keys until the final batch state has been validated.
            for name in db.query(EntityName).filter_by(entity_id=entity.id):
                name.claim_key = None
            db.flush()
            if op.action != 'create_entity':
                self._operate(db, entity, op, provenance, source_message_id, affected, before, initial_versions, expected_versions, manual)
                if op.decision_origin == 'DELEGATED':
                    prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key='canonical_name').first()
                    if prop and prop.fact_id:
                        revision = db.query(FactRevision).filter_by(fact_id=prop.fact_id).order_by(FactRevision.version_number.desc()).first()
                        if revision:
                            revision.changed_by = 'AI_AUTHOR_DELEGATED'
                            revision.change_reason = f'Author delegation: {op.source_quote} | Scope: {op.decision_target}'
            results.append({'index': index, 'entity_id': entity.id, 'entity_ref': entity.reference_label,
                'state': 'validated', 'action': op.action, 'property_key': op.property_key, 'value': op.value,
                'provenance':provenance, 'source_quote':op.source_quote})
        db.flush()
        self.validate(db, world_id, affected)
        batch.status = 'validated'
        for entity in affected.values():
            entity.version_number += 1
            changed_keys = {('canonical_name' if op.action == 'rename' else op.property_key) for op in operations
                            if (op.entity_id == entity.id or key(op.entity_ref) == entity.reference_key)}
            for issue in db.query(CanonSyncIssue).filter_by(entity_id=entity.id, status='OPEN'):
                if any(issue.detail.startswith(f'Property {field}:') for field in changed_keys):
                    issue.status = 'RESOLVED'
            db.flush()
            entity.canon_status = 'NEEDS_REVIEW' if db.query(CanonSyncIssue).filter_by(entity_id=entity.id, status='OPEN').first() else ACTIVE
            db.add(EntityRevision(entity_id=entity.id, batch_id=batch.id, version_number=entity.version_number,
                previous_json=dump(before[entity.id]), new_json=dump(self.snapshot(db, entity))))
        for row in results:
            if row['state'] == 'validated':
                row['state'] = 'committed'
        batch.status = 'proposed' if not affected else 'committed'
        result = {'batch_id': batch.id, 'status': batch.status, 'operations': results,
                  'entities': list(affected), 'clarifications': [],
                  'authorization':'AUTHOR_APPROVAL' if manual == 'APPROVED' else 'AUTHOR_EDITOR' if manual else 'AUTHOR_EVIDENCE'}
        proposed = [row['operation'] for row in results if row['state'] == 'proposed']
        if proposed:
            if not affected:
                batch.operations_json = dump(proposed)
            else:
                encoded = dump(proposed)
                child = CanonOperationBatch(world_id=world_id, idempotency_key=idempotency_key + ':proposals',
                    fingerprint=hashlib.sha256(encoded.encode()).hexdigest(), status='proposed', operations_json=encoded,
                    result_json=dump({'status':'proposed', 'operations':[row for row in results if row['state']=='proposed']}))
                db.add(child); db.flush(); result['proposal_batch_id'] = child.id
        for entity in affected.values():
            prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key='canonical_name').first()
            if prop and prop.canon_status == 'UNRESOLVED_POSSIBILITY':
                result['clarifications'].append(f'What canonical name should {entity.reference_label} use?')
        result['message'] = self.outcome(result, affected)
        batch.result_json = dump(result)
        db.flush()
        return result

    def _operate(self, db, entity, op, provenance, source_id, affected, before, versions, expected, manual):
        if op.action in {'rename', 'set_property', 'clear_property', 'mark_superseded'}:
            prop_key = 'canonical_name' if op.action == 'rename' else op.property_key
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,99}', prop_key):
                fail('Invalid property key.', 'INVALID_OPERATION')
            prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key=prop_key).first()
            if not prop:
                prop = EntityProperty(entity_id=entity.id, property_key=prop_key)
                db.add(prop)
            prop.value_json = dump(None if op.action == 'clear_property' else op.value)
            prop.provenance, prop.source_quote, prop.source_message_id = provenance, op.source_quote, source_id
            prop.canon_status = 'SUPERSEDED' if op.action == 'mark_superseded' else 'UNRESOLVED_POSSIBILITY' if op.action == 'clear_property' else ACTIVE
            prop.version_number = (prop.version_number or 0) + 1
            if prop_key == 'canonical_name':
                for old in db.query(EntityName).filter(EntityName.entity_id == entity.id, EntityName.kind == 'CANONICAL', EntityName.canon_status.in_([ACTIVE, 'NEEDS_REVIEW'])):
                    old.kind = 'FORMER'; old.canon_status = 'SUPERSEDED'; old.claim_key = None; old.version_number += 1
                if prop.canon_status == ACTIVE:
                    if provenance == 'AUTHOR_CREATED':
                        for rejected in db.query(EntityName).filter_by(entity_id=entity.id, kind='REJECTED', normalized_name=key(op.value), canon_status=ACTIVE):
                            rejected.canon_status = 'SUPERSEDED'; rejected.version_number += 1
                    self._name(db, entity, op.value, 'CANONICAL', provenance, op.source_quote, source_id)
            db.flush(); self._project(db, entity, prop, op.section_slug)
            if prop_key == 'canonical_name':
                self._project_names(db, entity)
        elif op.action in {'add_alias', 'reserve_name', 'reject_name'}:
            kind = {'add_alias': 'ALIAS', 'reserve_name': 'RESERVED', 'reject_name': 'REJECTED'}[op.action]
            self._name(db, entity, op.value, kind, provenance, op.source_quote, source_id)
            if op.action == 'reject_name':
                prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key='canonical_name').first()
                if prop and key(json.loads(prop.value_json)) == key(op.value):
                    clear = op.model_copy(update={'action': 'clear_property', 'property_key': 'canonical_name', 'value': None})
                    self._operate(db, entity, clear, provenance, source_id, affected, before, versions, expected, manual)
            self._project_names(db, entity)
        elif op.action in {'remove_alias', 'release_name'}:
            kinds = ['ALIAS'] if op.action == 'remove_alias' else ['RESERVED', 'REJECTED']
            for name in db.query(EntityName).filter(EntityName.entity_id == entity.id, EntityName.kind.in_(kinds), EntityName.normalized_name == key(op.value)):
                name.canon_status = 'SUPERSEDED'; name.claim_key = None; name.version_number += 1
            self._project_names(db, entity)
        elif op.action in {'add_constraint', 'remove_constraint'}:
            if op.constraint_type not in {'forbidden_name', 'immutable_property', 'property_equals'}:
                fail('This constraint needs an explicit supported interpretation.', 'CLARIFICATION_REQUIRED')
            existing = db.query(EntityConstraint).filter_by(entity_id=entity.id, constraint_type=op.constraint_type, value_json=dump(op.value)).first()
            if op.action == 'remove_constraint':
                if existing:
                    existing.canon_status = 'SUPERSEDED'
                    existing.version_number += 1
            elif not existing:
                db.add(EntityConstraint(world_id=entity.world_id, entity_id=entity.id, constraint_type=op.constraint_type,
                    value_json=dump(op.value), provenance=provenance, source_quote=op.source_quote, source_message_id=source_id))
            db.flush(); self._project_names(db, entity)
        elif op.action in {'add_relationship', 'remove_relationship'}:
            target = self.resolve(db, entity.world_id, op.related_entity_id, op.related_entity_ref, required=False)
            if target is None:
                target = self._create(db, entity.world_id, op.model_copy(update={
                    'entity_id':None, 'entity_ref':op.related_entity_ref,
                    'entity_type':op.related_entity_ref.split(' ',1)[0].casefold(), 'scope_key':None}), provenance, source_id)
                before[target.id], versions[target.id] = None, target.version_number
            version = expected.get(target.id) if expected.get('_strict_snapshot') else op.related_expected_version
            if target.id not in versions:
                if version != target.version_number:
                    fail('The related entity changed or needs an expected version.', 'STALE_ENTITY_VERSION')
                before[target.id] = self.snapshot(db, target); versions[target.id] = target.version_number
            affected[target.id] = target
            if op.relationship_type not in RELATIONS or target.id == entity.id:
                fail('Unsupported or self-referential relationship.', 'INVALID_RELATIONSHIP')
            relation = db.query(EntityRelationship).filter_by(source_entity_id=entity.id, target_entity_id=target.id, relationship_type=op.relationship_type).first()
            if relation is None and op.action == 'add_relationship':
                relation = EntityRelationship(world_id=entity.world_id, source_entity_id=entity.id, target_entity_id=target.id,
                    relationship_type=op.relationship_type, details_json=dump(op.value or {}), provenance=provenance,
                    source_quote=op.source_quote, source_message_id=source_id)
                db.add(relation)
            elif relation:
                relation.canon_status = 'SUPERSEDED' if op.action == 'remove_relationship' else ACTIVE
                relation.version_number += 1
                if op.action == 'add_relationship':
                    relation.details_json = dump(op.value or {})
                    relation.provenance, relation.source_quote, relation.source_message_id = provenance, op.source_quote, source_id
            db.flush(); self._project_names(db, entity)
        else:
            fail('Unsupported operation.', 'INVALID_OPERATION')

    def _name(self, db, entity, name, kind, provenance, quote, source_id):
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            fail('A name must be a nonempty string of at most 200 characters.', 'INVALID_NAME')
        row = db.query(EntityName).filter_by(entity_id=entity.id, kind=kind, normalized_name=key(name), canon_status=ACTIVE).first()
        if row is None:
            row = EntityName(world_id=entity.world_id, entity_id=entity.id, name=name.strip(), normalized_name=key(name),
                kind=kind, scope_key=entity.scope_key, provenance=provenance, source_quote=quote, source_message_id=source_id)
            db.add(row)
        return row

    def validate(self, db, world_id, affected):
        """Bounded dependency validation and scoped claim checks before any commit."""
        db.flush()
        for row in db.query(EntityName).filter(EntityName.entity_id.in_(affected)):
            row.claim_key = None
        db.flush()
        assigned_claims = set()
        for entity in affected.values():
            values = {p.property_key: json.loads(p.value_json) for p in db.query(EntityProperty).filter_by(entity_id=entity.id, canon_status=ACTIVE)}
            for field in ('birth_year', 'death_year', 'age', 'year'):
                if field in values and (isinstance(values[field], bool) or not isinstance(values[field], (int, float)) or not math.isfinite(values[field])):
                    fail(f'Confirm a numeric year/age for {entity.reference_label}.', 'CLARIFICATION_REQUIRED')
            for name in db.query(EntityName).filter_by(entity_id=entity.id, canon_status=ACTIVE):
                if name.kind == 'CANONICAL' and db.query(EntityName).filter_by(entity_id=entity.id, kind='REJECTED', normalized_name=name.normalized_name, canon_status=ACTIVE).first():
                    fail(f'{name.name} is rejected for {entity.reference_label}. Remove that restriction explicitly first.')
            for constraint in db.query(EntityConstraint).filter_by(entity_id=entity.id, canon_status=ACTIVE):
                value = json.loads(constraint.value_json)
                if constraint.constraint_type == 'forbidden_name' and key(values.get('canonical_name')) == key(value):
                    fail(f'That name is prohibited for {entity.reference_label}.')
                if constraint.constraint_type in {'immutable_property', 'property_equals'}:
                    if not isinstance(value, dict) or values.get(value.get('property_key')) != value.get('value'):
                        fail(f'A protected property of {entity.reference_label} would change.')
            if values.get('birth_year') is not None and values.get('death_year') is not None and values['birth_year'] > values['death_year']:
                fail('Birth must not follow death.')
            if values.get('age') is not None and (not isinstance(values['age'], (int, float)) or values['age'] < 0):
                fail('Age must be a nonnegative number.')
            policy = db.query(NamingPolicy).filter_by(world_id=world_id, entity_type=entity.entity_type).first()
            unique = policy.canonical_unique if policy else entity.entity_type not in {'character', 'family', 'species'}
            aliases_unique = policy.aliases_unique if policy else False
            for name in db.query(EntityName).filter_by(entity_id=entity.id, canon_status=ACTIVE):
                if name.kind == 'CANONICAL':
                    conflict_kinds = ['RESERVED'] + (['CANONICAL'] if unique else []) + (['ALIAS'] if aliases_unique else [])
                    claim_required = unique
                elif name.kind == 'RESERVED':
                    conflict_kinds = ['CANONICAL', 'RESERVED'] + (['ALIAS'] if aliases_unique else [])
                    claim_required = True
                elif name.kind == 'ALIAS' and aliases_unique:
                    conflict_kinds, claim_required = ['CANONICAL', 'RESERVED', 'ALIAS'], True
                else:
                    continue
                conflicts = db.query(EntityName).join(Entity, Entity.id == EntityName.entity_id).filter(
                    EntityName.world_id == world_id, Entity.entity_type == entity.entity_type,
                    EntityName.scope_key == entity.scope_key, EntityName.normalized_name == name.normalized_name,
                    EntityName.entity_id != entity.id, EntityName.canon_status.in_([ACTIVE, 'NEEDS_REVIEW']),
                    EntityName.kind.in_(conflict_kinds)).all()
                if conflicts:
                    fail(f'{name.name} is assigned or reserved for another {entity.entity_type} in this scope. No changes were saved.')
                if unique or name.kind != 'CANONICAL':
                    self._check_legacy_claims(db, entity, name.name)
                if not claim_required:
                    continue
                claim = hashlib.sha256(dump([world_id, entity.entity_type, entity.scope_key, name.normalized_name]).encode()).hexdigest()
                same_owner = db.query(EntityName).filter_by(entity_id=entity.id, claim_key=claim).first()
                if claim not in assigned_claims and (not same_owner or same_owner.id == name.id):
                    name.claim_key = claim
                    assigned_claims.add(claim)
        relations = db.query(EntityRelationship).filter(EntityRelationship.world_id == world_id,
            EntityRelationship.canon_status == ACTIVE, or_(EntityRelationship.source_entity_id.in_(affected),
            EntityRelationship.target_entity_id.in_(affected))).all()
        def props(eid):
            return {p.property_key: json.loads(p.value_json) for p in db.query(EntityProperty).filter_by(entity_id=eid, canon_status=ACTIVE)}
        for relation in relations:
            source, target = props(relation.source_entity_id), props(relation.target_entity_id)
            if relation.relationship_type == 'parent_of':
                if 'birth_year' in source and 'birth_year' in target and source['birth_year'] >= target['birth_year']:
                    fail('A biological parent must be born before the child.')
                if 'age' in source and 'age' in target and source['age'] <= target['age']:
                    fail('The proposed ages conflict with the parent relationship.')
                if not ({'birth_year'} <= source.keys() & target.keys() or {'age'} <= source.keys() & target.keys()):
                    fail('Confirm compatible parent/child chronology before establishing this relationship.', 'CLARIFICATION_REQUIRED')
            elif relation.relationship_type == 'participates_in':
                year = target.get('year')
                if year is None or ('birth_year' not in source):
                    fail('Confirm the event year and participant chronology.', 'CLARIFICATION_REQUIRED')
                if year < source['birth_year'] or ('death_year' in source and year > source['death_year']):
                    fail('The participant is not alive at the established event date.')
            elif relation.relationship_type == 'treaty_with':
                details = json.loads(relation.details_json)
                for eid, values in ((relation.source_entity_id, source), (relation.target_entity_id, target)):
                    for prop, expected in details.get('protected_properties', {}).get(eid, {}).items():
                        if values.get(prop) != expected:
                            fail('The proposed change conflicts with an established treaty dependency.')

    def _check_legacy_claims(self, db, entity, name):
        # Targeted compatibility lookup; do not silently ignore unconverted owners.
        from backend.services.canon_language import REF
        pattern = re.compile(rf"^({REF})(?:'s name is| is (?:called |named )?){re.escape(name)}(?!\w)(?=[.,;]|\s+(?:and|which|closest|orbits)\b|$)", re.I)
        candidates = [fact.content for fact in db.query(Fact).filter(Fact.world_id == entity.world_id,
            Fact.canon_status == ACTIVE, Fact.content.ilike('%' + name + '%'))]
        candidates += [block.strip().lstrip('-* ') for section in db.query(BibleSection).filter(
            BibleSection.world_id == entity.world_id, BibleSection.markdown_content.ilike('%' + name + '%'))
            for block in section.markdown_content.split('\n\n')]
        for statement in candidates:
            match = pattern.match(statement)
            if match and key(match[1]) != entity.reference_key and match[1].split()[0].lower() == entity.entity_type:
                owner = self.resolve(db, entity.world_id, ref=match[1], required=False)
                if owner is None or owner.scope_key == entity.scope_key:
                    fail(f'{name} already belongs to {match[1]} in existing World Bible text. Confirm the ownership change first.')

    def _project(self, db, entity, prop, section_slug=''):
        if not projection_enabled.get():
            return
        from backend.services.consistency_service import consistency_service
        value = json.loads(prop.value_json)
        content = (f'{entity.reference_label} is called {value}.' if prop.property_key == 'canonical_name' and value is not None
                   else f"{entity.reference_label}'s {prop.property_key.replace('_', ' ')} is {value}." if value is not None
                   else f'{entity.reference_label}: {prop.property_key} is unresolved; no replacement has been chosen.')
        fact = db.get(Fact, prop.fact_id) if prop.fact_id else None
        old = fact.content if fact else ''
        if fact is None:
            fact = Fact(world_id=entity.world_id, title=f'{entity.reference_label}: {prop.property_key}', content=content,
                provenance_type=prop.provenance, canon_status=prop.canon_status, source_message_id=prop.source_message_id,
                authority_tier=1 if prop.provenance == 'AUTHOR_CREATED' else 2, version_number=1)
            db.add(fact); db.flush(); prop.fact_id = fact.id
            db.add(FactRevision(fact_id=fact.id, version_number=1, new_content=content,
                new_canon_status=fact.canon_status, new_provenance_type=fact.provenance_type,
                changed_by='AUTHOR_CHAT_CONFIRMATION', change_reason='Structured property created'))
        else:
            previous = (fact.content, fact.canon_status, fact.provenance_type)
            fact.content, fact.canon_status, fact.provenance_type = content, prop.canon_status, prop.provenance
            fact.version_number += 1
            db.add(FactRevision(fact_id=fact.id, version_number=fact.version_number, previous_content=previous[0],
                new_content=content, previous_canon_status=previous[1], new_canon_status=fact.canon_status,
                previous_provenance_type=previous[2], new_provenance_type=fact.provenance_type,
                changed_by='AUTHOR_CHAT_CONFIRMATION', change_reason='Structured property update'))
        consistency_service.sync_fact_fts(db, fact)
        links = db.query(BibleSectionFact).filter_by(fact_id=fact.id).all()
        changed = False
        for link in links:
            section = db.get(BibleSection, link.section_id)
            if old and section.markdown_content.count(old) == 1:
                self._section_write(db, section, section.markdown_content.replace(old, content, 1))
                changed = True
            elif old:
                fail('The linked prose changed. Resolve its synchronization issue before overwriting it.', 'MARKDOWN_SYNC_CONFLICT')
        if not changed:
            section = db.query(BibleSection).filter_by(world_id=entity.world_id,
                slug=section_slug or TYPE_SECTIONS.get(entity.entity_type, 'cultures-societies')).first()
            if section:
                self._section_write(db, section, section.markdown_content.rstrip() + '\n\n' + content)
                db.add(BibleSectionFact(section_id=section.id, fact_id=fact.id))
        db.flush()

    def _render_names(self, db, entity, state):
        labels = {'ALIAS':'Alias', 'FORMER':'Former name', 'RESERVED':'Reserved name', 'REJECTED':'Rejected name'}
        lines = []
        for row in sorted(state['names'], key=lambda item:(item['kind'],item['name'])):
            if row['kind'] == 'CANONICAL':
                continue
            label = labels.get(row['kind'], row['kind'].replace('_',' ').title())
            if row['canon_status'] == 'SUPERSEDED' and row['kind'] != 'FORMER':
                label = 'Former ' + label.lower()
            source = 'author decision' if row['provenance'] == 'AUTHOR_CREATED' else 'author-approved AI'
            lines.append(f'- {label}: {row["name"]} ({source})')
        for row in state['constraints']:
            if row['canon_status'] == ACTIVE:
                lines.append(f'- Constraint: {row["constraint_type"].replace("_", " ")} {row["value_json"]}')
        for row in state['relationships']:
            if row['canon_status'] != ACTIVE or row['source_entity_id'] != entity.id:
                continue
            target = db.get(Entity, row['target_entity_id'])
            lines.append(f'- {row["relationship_type"].replace("_", " ").capitalize()}: {target.reference_label}')
        return ('### ' + entity.reference_label + ' — Names and constraints\n' + '\n'.join(lines)) if lines else ''

    def _project_names(self, db, entity):
        if not projection_enabled.get():
            return
        db.flush()
        block = self._render_names(db, entity, self.snapshot(db, entity))
        cache = db.info.setdefault('entity_projection_slots', {})
        previous = cache.get(entity.id)
        if previous is None:
            revision = db.query(EntityRevision).filter_by(entity_id=entity.id).order_by(EntityRevision.version_number.desc()).first()
            state = json.loads(revision.new_json) if revision else {}
            previous = self._render_names(db, entity, state) if state and state.get('names_projected', True) else ''
        section = db.query(BibleSection).filter_by(world_id=entity.world_id, slug=TYPE_SECTIONS.get(entity.entity_type, 'cultures-societies')).first()
        if section is None:
            return
        title = '### ' + entity.reference_label + ' — Names and constraints'
        if previous:
            if section.markdown_content.count(previous) != 1:
                fail('The author-edited names/constraints text needs clarification before synchronization.', 'MARKDOWN_SYNC_CONFLICT')
            content = section.markdown_content.replace(previous, block, 1)
        elif title in section.markdown_content:
            fail('An author-written heading overlaps the generated entity view. Confirm its interpretation.', 'MARKDOWN_SYNC_CONFLICT')
        else:
            content = section.markdown_content.rstrip() + ('\n\n' + block if block else '')
        self._section_write(db, section, content)
        cache[entity.id] = block

    def _section_write(self, db, section, content):
        if section.markdown_content == content:
            return
        old = section.markdown_content
        section.markdown_content = content; section.version_number += 1
        db.add(BibleSectionRevision(section_id=section.id, version_number=section.version_number,
            previous_markdown=old, new_markdown=content, edit_origin='STRUCTURED_CANON', author_note='Validated entity operation'))

    def _manual_noop(self, db, world_id, operation):
        entity = self.resolve(db, world_id, operation.entity_id, operation.entity_ref, required=False)
        if entity and operation.action in {'rename','set_property'}:
            field = 'canonical_name' if operation.action == 'rename' else operation.property_key
            prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key=field, canon_status=ACTIVE).first()
            return prop is not None and json.loads(prop.value_json) == operation.value
        return False

    def sync_manual_section(self, db, section, old_markdown):
        """Preserve direct prose; uncertain structured synchronization is explicit."""
        from backend.schemas import AuthorDecisionSchema
        from backend.services.canon_language import compile_decisions
        paragraphs = [p.strip() for p in section.markdown_content.split('\n\n') if p.strip() and not p.startswith('#')]
        operations = compile_decisions([AuthorDecisionSchema(kind='DECLARATION', source_quote=p) for p in paragraphs], section.markdown_content)
        tracked = db.query(EntityProperty).join(Entity).join(BibleSectionFact, BibleSectionFact.fact_id == EntityProperty.fact_id).filter(
            Entity.world_id == section.world_id, BibleSectionFact.section_id == section.id).all()
        for op in operations:
            entity = self.resolve(db, section.world_id, ref=op.entity_ref, required=False)
            if entity:
                op.entity_id, op.expected_version = entity.id, entity.version_number
        operations = [op for op in operations if not self._manual_noop(db, section.world_id, op)]
        token = projection_enabled.set(False)
        try:
            try:
                with db.begin_nested():
                    if operations:
                        self.apply(db, section.world_id, operations,
                            f'manual:{section.id}:{section.version_number}', manual=True)
                        for op in operations:
                            entity = self.resolve(db, section.world_id, op.entity_id, op.entity_ref)
                            prop_key = 'canonical_name' if op.action == 'rename' else op.property_key
                            prop = db.query(EntityProperty).filter_by(entity_id=entity.id, property_key=prop_key).first()
                            source = db.query(Fact).join(BibleSectionFact).filter(BibleSectionFact.section_id == section.id,
                                Fact.content == op.source_quote.lstrip('-* ')).first()
                            if prop and source:
                                prop.fact_id = source.id
                    # Mark unsupported edits to tracked statements rather than inventing values.
                    covered = {(key(op.entity_ref), 'canonical_name' if op.action == 'rename' else op.property_key) for op in operations}
                    for prop in tracked:
                        entity = db.get(Entity, prop.entity_id)
                        if (entity.reference_key, prop.property_key) not in covered and old_markdown != section.markdown_content:
                            fact = db.get(Fact, prop.fact_id)
                            rendered = (f'{entity.reference_label} is called {json.loads(prop.value_json)}.' if prop.property_key == 'canonical_name'
                                        else f"{entity.reference_label}'s {prop.property_key.replace('_', ' ')} is {json.loads(prop.value_json)}.")
                            if rendered and rendered in old_markdown and rendered not in section.markdown_content:
                                self._sync_issue(db, section, entity, 'The author edited a tracked property in prose; confirm its structured interpretation.', prop.property_key)
            except LocalModelError as exc:
                db.expire_all()
                section = db.get(BibleSection, section.id)
                for op in operations:
                    entity = self.resolve(db, section.world_id, ref=op.entity_ref, required=False)
                    self._sync_issue(db, section, entity, exc.message, 'canonical_name' if op.action == 'rename' else op.property_key)
        finally:
            projection_enabled.reset(token)

    def _sync_issue(self, db, section, entity, detail, property_key=''):
        db.add(CanonSyncIssue(world_id=section.world_id, entity_id=entity.id if entity else None,
            section_id=section.id, detail=f'Property {property_key}: ' + detail))
        section.has_pending_review = True
        ids = [entity.id] if entity else []
        for eid in set(ids):
            owner = db.get(Entity, eid); owner.version_number += 1; owner.canon_status = 'NEEDS_REVIEW'
            for prop in db.query(EntityProperty).filter_by(entity_id=eid, property_key=property_key):
                prop.canon_status = 'UNRESOLVED_POSSIBILITY'
                if prop.fact_id:
                    fact = db.get(Fact, prop.fact_id)
                    if fact:
                        fact.consistency_status = 'NEEDS_REVIEW'
                        fact.canon_status = 'UNRESOLVED_POSSIBILITY'
            if property_key == 'canonical_name':
                for name in db.query(EntityName).filter_by(entity_id=eid, kind='CANONICAL'):
                    name.canon_status = 'NEEDS_REVIEW'; name.claim_key = None

    def execute(self, db, world_id, operations, idempotency_key, approval_batch_id=None, **kwargs):
        """Standalone endpoint entry; real rollback includes all projection writes."""
        try:
            if db.in_transaction():
                db.rollback()  # endpoints provide a fresh read-only session
            if db.bind.dialect.name == 'sqlite':
                db.execute(text('BEGIN IMMEDIATE'))
            approved = db.get(CanonOperationBatch, approval_batch_id) if approval_batch_id else None
            if approval_batch_id and (approved is None or approved.world_id != world_id):
                fail('Proposal not found.', 'INVALID_PROPOSAL')
            if approved and approved.status == 'approved':
                result = json.loads(approved.result_json)
                db.rollback()
                return result
            result = self.apply(db, world_id, operations, idempotency_key, **kwargs)
            if approved:
                approved.status, approved.result_json = 'approved', dump(result)
            db.commit()
            return result
        except Exception as exc:
            db.rollback()
            try:
                existing = db.query(CanonOperationBatch).filter_by(world_id=world_id, idempotency_key=idempotency_key).first()
                if existing is None and db.get(World, world_id):
                    payload = [op.model_dump() for op in operations]
                    status = 'clarification_required' if isinstance(exc, LocalModelError) and exc.error_code in {'CLARIFICATION_REQUIRED', 'AMBIGUOUS_ENTITY'} else 'failed'
                    db.add(CanonOperationBatch(world_id=world_id, idempotency_key=idempotency_key,
                        fingerprint=hashlib.sha256(dump(payload).encode()).hexdigest(), status=status, operations_json=dump(payload),
                        result_json=dump({'status': status, 'message': str(exc), 'operations': [{'index': i, 'state': status} for i in range(len(operations))]})))
                    db.commit()
            except Exception:
                db.rollback()
            raise

    @staticmethod
    def outcome(result, affected):
        if result['status'] == 'proposed':
            return "The change is a proposal. It has not been added to established canon."
        labels = ', '.join(entity.reference_label for entity in affected.values())
        details = []
        for op in result['operations']:
            if op['state'] != 'committed':
                continue
            if op['action'] == 'rename':
                details.append(f"{op['entity_ref']}'s canonical name is now {op['value']}.")
            elif op['action'] in {'reserve_name', 'reject_name', 'add_alias'}:
                details.append(f"{op['entity_ref']}: {op['action'].replace('_', ' ')} {op['value']} committed.")
        return (f'World Bible changes committed for {labels}. ' + ' '.join(details)
                + (' Additional changes remain unapproved proposals.' if any(op['state']=='proposed' for op in result['operations']) else '')
                + (' ' + ' '.join(result['clarifications']) if result['clarifications'] else '')).strip()


entity_service = EntityService()
