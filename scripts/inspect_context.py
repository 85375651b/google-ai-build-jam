"""Read-only development inspection; no inference, writes, or external research.

Example: .venv/bin/python scripts/inspect_context.py --world-id UUID
         --conversation-id UUID --message "Help me name Planet X"
Add --include-text explicitly to display selected unpublished content locally.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database import SessionLocal
from backend.models import Conversation, World
from backend.services.context_retrieval import context_retriever, estimate_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-id', required=True)
    parser.add_argument('--conversation-id')
    parser.add_argument('--message', required=True)
    parser.add_argument('--include-text', action='store_true')
    args = parser.parse_args()
    with SessionLocal() as db:
        if db.get(World, args.world_id) is None:
            parser.error('World not found.')
        conversation = db.get(Conversation, args.conversation_id) if args.conversation_id else None
        if args.conversation_id and (conversation is None or conversation.world_id != args.world_id):
            parser.error('Conversation does not belong to this world.')
        bundle = context_retriever.retrieve(db, args.world_id, args.message, conversation=conversation)
        selected = []
        for item in bundle.items:
            record = dict(kind=item.kind, id=item.id, reason=item.reason, score=round(item.score, 2),
                          essential=item.essential, estimated_tokens=estimate_tokens(item.text), **item.details)
            if args.include_text:
                record['text'] = item.text
            selected.append(record)
        output = {'task_category': bundle.category, 'entities': sorted(bundle.entities),
                  'related_entities': sorted(bundle.related_entities), 'selected': selected,
                  'recent_message_ids': [message.id for message in bundle.recent],
                  'summary_included': bool(bundle.summary), 'reductions': bundle.reductions}
        if args.include_text:
            output['summary'] = bundle.summary
            output['recent'] = [{'role': message.role, 'text': message.content} for message in bundle.recent]
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
