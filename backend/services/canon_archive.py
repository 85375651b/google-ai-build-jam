"""Additive project-archive extension; remap stable internal references on copy-import."""
from datetime import datetime
import json
from uuid import uuid4
from backend import models as m

TABLE_MODELS = [m.Entity, m.EntityProperty, m.EntityName, m.EntityRelationship, m.EntityConstraint,
                m.EntityRevision, m.CanonOperationBatch, m.NamingPolicy, m.CanonSyncIssue]


def export_canon(db, world_id):
    entity_ids = [e.id for e in db.query(m.Entity).filter_by(world_id=world_id)]
    fact_ids = [f.id for f in db.query(m.Fact).filter_by(world_id=world_id)]
    section_ids = [s.id for s in db.query(m.BibleSection).filter_by(world_id=world_id)]
    def encode(rows):
        return [{column.name: (getattr(row, column.name).isoformat() if isinstance(getattr(row, column.name), datetime)
                 else getattr(row, column.name)) for column in row.__table__.columns} for row in rows]
    result = {'version': 1}
    for model in TABLE_MODELS:
        query = db.query(model)
        query = query.filter(model.world_id == world_id) if hasattr(model, 'world_id') else query.filter(model.entity_id.in_(entity_ids))
        result[model.__tablename__] = encode(query.all())
    result['fact_revisions'] = encode(db.query(m.FactRevision).filter(m.FactRevision.fact_id.in_(fact_ids)))
    result['bible_section_facts'] = encode(db.query(m.BibleSectionFact).filter(m.BibleSectionFact.section_id.in_(section_ids)))
    result['fact_citations'] = []
    for citation in db.query(m.FactCitation).filter(m.FactCitation.fact_id.in_(fact_ids)):
        row = encode([citation])[0]
        chunk = db.get(m.ReferenceChunk, citation.reference_chunk_id) if citation.reference_chunk_id else None
        source = db.get(m.ResearchSource, citation.research_source_id) if citation.research_source_id else None
        row['locator'] = ({'chapter':chunk.chapter_title,'section':chunk.section_title,
                           'book_page':chunk.book_page_start,'pdf_page':chunk.pdf_page_start} if chunk else
                          {'url':source.url,'title':source.title} if source else {})
        result['fact_citations'].append(row)
    return result


def import_canon(db, world, extension, mapping):
    if not extension:
        return
    for model in TABLE_MODELS:
        for row in extension.get(model.__tablename__, []):
            mapping[row['id']] = str(uuid4())
    new_entity_ids = {mapping[row['id']] for row in extension.get('entities', [])}
    def remap(value):
        if isinstance(value, dict):
            return {mapping.get(k, k): remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, str):
            if value in mapping:
                return mapping[value]
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(remap(parsed), ensure_ascii=False)
            except ValueError:
                pass
        return value
    for model in TABLE_MODELS:
        for original in extension.get(model.__tablename__, []):
            row = {column.name: remap(original[column.name]) for column in model.__table__.columns if column.name in original}
            if 'world_id' in row:
                row['world_id'] = world.id
            for field in ('entity_id', 'source_entity_id', 'target_entity_id'):
                if field in row and row[field] is not None and row[field] not in new_entity_ids:
                    raise ValueError('Structured archive references an entity outside the imported world.')
            if model is m.EntityName:
                row['claim_key'] = None  # Rebuild with the new world's namespace.
            if model is m.CanonOperationBatch:
                import hashlib
                from backend.services.entity_service import dump
                row['fingerprint'] = hashlib.sha256(dump(json.loads(row['operations_json'])).encode()).hexdigest()
            for column in model.__table__.columns:
                if isinstance(row.get(column.name), str) and column.type.python_type is datetime:
                    row[column.name] = datetime.fromisoformat(row[column.name])
            db.add(model(**row))
        db.flush()
    # The existing importer supplies a synthetic Fact revision. Replace it with
    # the original history when the new archive extension contains that history.
    source_revisions = extension.get('fact_revisions', [])
    restored = {mapping[row['fact_id']] for row in source_revisions if row['fact_id'] in mapping}
    if restored:
        db.query(m.FactRevision).filter(m.FactRevision.fact_id.in_(restored)).delete(synchronize_session=False)
    for original in source_revisions:
        row = dict(original, id=str(uuid4()), fact_id=mapping[original['fact_id']])
        if row.get('created_at'):
            row['created_at'] = datetime.fromisoformat(row['created_at'])
        db.add(m.FactRevision(**row))
    for original in extension.get('bible_section_facts', []):
        row = remap(original)
        if db.get(m.BibleSectionFact, (row['section_id'], row['fact_id'])) is None:
            db.add(m.BibleSectionFact(**row))
    for original in extension.get('fact_citations', []):
        locator = original.get('locator', {})
        chunk = db.get(m.ReferenceChunk, original.get('reference_chunk_id')) if original.get('reference_chunk_id') else None
        source = db.query(m.ResearchSource).filter_by(url=locator.get('url')).first() if locator.get('url') else None
        note = original.get('citation_note') or ''
        if locator and not chunk and not source:
            note += '\nImported source locator (reference not indexed here): ' + json.dumps(locator)
        db.add(m.FactCitation(id=str(uuid4()), fact_id=mapping[original['fact_id']],
            citation_source_type=original['citation_source_type'], reference_chunk_id=chunk.id if chunk else None,
            research_source_id=source.id if source else None, citation_note=note))
    # Generated prose markers contain stable IDs and must follow the new identity map.
    for section in db.query(m.BibleSection).filter_by(world_id=world.id):
        for old, new in mapping.items():
            section.markdown_content = section.markdown_content.replace(f'entity:{old}:', f'entity:{new}:').replace(f'[entity {old}]', f'[entity {new}]')
    db.flush()
    # Imported data is preserved; incompatible legacy claims are explicitly reviewable.
    from backend.services.entity_service import entity_service
    try:
        with db.begin_nested():
            entity_service.validate(db, world.id, {e.id: e for e in db.query(m.Entity).filter_by(world_id=world.id)})
            db.flush()
    except Exception:
        db.add(m.CanonSyncIssue(world_id=world.id, detail='Imported structured constraints need author review. Imported content was preserved.'))
