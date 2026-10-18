// SPDX-License-Identifier: AGPL-3.0-only
// Synthetic interaction data only. Never included in the maintained dashboard.
export const fixtureRecord = {
  manifest: {
    schemaVersion: 2, title: 'Synthetic dashboard checks', description: 'Interaction fixtures, not project progress.',
    heroes: [{id: 'sample', file: 'heroes/01-sample.json'}, {id: 'other', file: 'heroes/02-other.json'}],
  },
  heroes: [
    {
      schemaVersion: 2, id: 'sample', number: 1, title: 'Sample Hero', summary: 'Synthetic navigation and state checks.', updatedAt: '2026-10-18',
      goals: [
        {id: 'sample-first', title: 'First sample Goal', milestones: [
          {id: 'sample-done', title: 'Completed sample', status: 'done', description: 'Synthetic evidence text.',
            acceptance: ['The sample criterion remains readable.'], evidence: [{label: 'Fixture source', path: 'web/dashboard/tests/fixtures.mjs'}]},
          {id: 'sample-active', title: 'Active sample', status: 'in_progress'},
          {id: 'sample-ready', title: 'Ready sample', status: 'ready'},
        ]},
        {id: 'sample-second', title: 'Second sample Goal', milestones: [
          {id: 'sample-blocked', title: 'Blocked sample', status: 'blocked', description: 'Waiting on the other sample.', dependsOn: ['other-done']},
          {id: 'sample-proposed', title: 'Proposed sample', status: 'proposed'},
        ]},
        {id: 'sample-empty', title: 'Empty sample Goal', milestones: []},
      ],
    },
    {
      schemaVersion: 2, id: 'other', number: 2, title: 'Other sample Hero', summary: 'Synthetic dependency destination.', updatedAt: '2026-10-18',
      goals: [{id: 'other-goal', title: 'Other sample Goal', milestones: [
        {id: 'other-done', title: 'Dependency sample', status: 'done', acceptance: ['The dependency is navigable.'],
          evidence: [{label: 'Fixture source', path: 'web/dashboard/tests/fixtures.mjs'}]},
      ]}],
    },
  ],
};
