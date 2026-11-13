"""Synthetic, non-manuscript context-size scenarios shared by tests and benchmarks."""
from datetime import datetime, timedelta, timezone

from backend.models import BibleSection, Conversation, Fact, Message

SCENARIOS = {
    'short naming correction': "Planet X is not Enceladus. That's Planet Y's name.",
    'factual research': 'How do rain shadows affect monsoon climate?',
    'character development': "Help me develop Lianne's established motivations and family relationships.",
    'political worldbuilding': 'Help me think through the political relationship between the Northern Kingdom and the Coastal Republic.',
    'cross-entity consistency': "Would naming Planet X Enceladus conflict with Planet Y's reservation?",
    'historical conversation': "Remember why Lianne's mother chose that name?",
}


def seed_context_scenarios(db, world, conversation):
    filler = 'Ceramic glazes exhibit cobalt flecking after kiln firing; artisans measure porosity, glaze thickness and vitrification. '
    for section in db.query(BibleSection).filter_by(world_id=world.id).order_by(BibleSection.display_order):
        section.markdown_content = '\n\n'.join(f'Archive tablet {section.display_order}-{i}: {filler * 3}' for i in range(16))
    relevant = [
        ('X orbit', 'Planet X orbits Helion and has a rocky surface.'),
        ('Y name', 'Enceladus is reserved for Planet Y. Planet X must not use that name.'),
        ('Lianne', 'Lianne is the harbor master. Her established motivation is protecting her family.'),
        ('Northern Kingdom', 'The Northern Kingdom and the Coastal Republic signed the Harbor Treaty.'),
        ('Coastal Republic', 'The Coastal Republic elects its council and trades grain with the Northern Kingdom.'),
    ]
    for index, (title, content) in enumerate(relevant):
        db.add(Fact(world_id=world.id, title=title, content=content,
                    updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)))
    for index in range(30):
        db.add(Fact(world_id=world.id, title=f'Ceramic ledger {index}', content=filler * 4))
    cosmos = db.query(BibleSection).filter_by(world_id=world.id, slug='cosmology-physics').one()
    cosmos.markdown_content += '\n\n### Inner planets\n\nPlanet X orbits Helion. Enceladus is reserved for Planet Y and cannot be used for Planet X.'
    culture = db.query(BibleSection).filter_by(world_id=world.id, slug='cultures-societies').one()
    culture.markdown_content += '\n\nLianne protects her family and runs the harbor.'
    politics = db.query(BibleSection).filter_by(world_id=world.id, slug='politics-governance').one()
    politics.markdown_content += '\n\nThe Northern Kingdom and Coastal Republic maintain the Harbor Treaty.'
    old = Conversation(world_id=world.id, title='Old naming discussion', is_archived=True)
    db.add(old); db.flush()
    db.add(Message(conversation_id=old.id, role='USER',
        content="Lianne's mother chose her name to honor her grandmother, a rescue captain.",
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
    for index in range(20):
        other = Conversation(world_id=world.id, title=f'Later ceramic discussion {index}')
        db.add(other); db.flush()
        db.add(Message(conversation_id=other.id, role='USER', content=filler))
    for index in range(10):
        db.add(Message(conversation_id=conversation.id, role='USER' if index % 2 == 0 else 'ASSISTANT',
                       content=f'Pottery exchange {index}: {filler * 12}'))
    db.add_all([Message(conversation_id=conversation.id, role='USER', content="Let us return to Lianne's background."),
                Message(conversation_id=conversation.id, role='ASSISTANT', content='We can refer to her established family relationships.')])
    conversation.rolling_summary = 'Established: Lianne is the harbor master. Unresolved: whether her family relocates.'
    db.commit()
