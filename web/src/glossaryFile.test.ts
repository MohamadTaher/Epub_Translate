import { describe, expect, it } from 'vitest';
import {
  describeImport,
  mergeGlossary,
  parseGlossaryFile,
  planImport,
  withLearned,
  type MergeReport,
} from '@/glossaryFile';

const report = (fields: Partial<MergeReport>): MergeReport => ({
  terms: {},
  added: 0,
  kept: 0,
  differed: 0,
  ...fields,
});

describe('parseGlossaryFile', () => {
  it('reads a glossary and trims what it finds', () => {
    expect(parseGlossaryFile('{" 道玄 ": " Dao Xuan "}')).toEqual({ 道玄: 'Dao Xuan' });
  });

  it('keeps a term whose translation has not been decided yet', () => {
    expect(parseGlossaryFile('{"道玄": ""}')).toEqual({ 道玄: '' });
  });

  it('drops a blank term, which is a row someone left behind', () => {
    expect(parseGlossaryFile('{"  ": "Dao Xuan", "青云": "Qingyun"}')).toEqual({ 青云: 'Qingyun' });
  });

  it('reads an empty glossary as an empty glossary', () => {
    expect(parseGlossaryFile('{}')).toEqual({});
  });

  it('explains a file that is not JSON at all', () => {
    expect(() => parseGlossaryFile('道玄 = Dao Xuan')).toThrow(/isn’t JSON/);
  });

  it('refuses a list, which is the likeliest wrong file', () => {
    expect(() => parseGlossaryFile('["道玄"]')).toThrow(/object of "term": "translation"/);
    expect(() => parseGlossaryFile('null')).toThrow(/object of "term": "translation"/);
  });

  it('names the term whose translation is not text', () => {
    expect(() => parseGlossaryFile('{"道玄": 3}')).toThrow(/道玄/);
  });
});

describe('mergeGlossary', () => {
  it('brings in what is new', () => {
    const result = mergeGlossary({ 道玄: 'Dao Xuan' }, { 青云: 'Qingyun' });
    expect(result.terms).toEqual({ 道玄: 'Dao Xuan', 青云: 'Qingyun' });
    expect(result).toMatchObject({ added: 1, kept: 0, differed: 0 });
  });

  it('keeps what this book already settled, and counts the disagreement', () => {
    const result = mergeGlossary({ Ivan: 'Ivan' }, { Ivan: 'John' });
    expect(result.terms).toEqual({ Ivan: 'Ivan' });
    expect(result).toMatchObject({ added: 0, kept: 1, differed: 1 });
  });

  it('does not call an identical translation a disagreement', () => {
    expect(mergeGlossary({ Ivan: 'Ivan' }, { Ivan: 'Ivan' })).toMatchObject({
      added: 0,
      kept: 1,
      differed: 0,
    });
  });

  it('does not call a blank in the file a disagreement either', () => {
    expect(mergeGlossary({ Ivan: 'Ivan' }, { Ivan: '' })).toMatchObject({ kept: 1, differed: 0 });
  });

  it('will not add a case variant of a term already here', () => {
    const result = mergeGlossary({ Ivan: 'Ivan' }, { IVAN: 'Johnny' });
    expect(result.terms).toEqual({ Ivan: 'Ivan' });
    expect(result).toMatchObject({ kept: 1, differed: 1 });
  });

  it('nor two case variants of a term the same file brought twice', () => {
    const result = mergeGlossary({}, { Ivan: 'Ivan', IVAN: 'Johnny' });
    expect(result.terms).toEqual({ Ivan: 'Ivan' });
    expect(result).toMatchObject({ added: 1, kept: 1 });
  });

  it('fills in a term left with no translation', () => {
    const result = mergeGlossary({ 道玄: '' }, { 道玄: 'Dao Xuan' });
    expect(result.terms).toEqual({ 道玄: 'Dao Xuan' });
    expect(result).toMatchObject({ added: 1, differed: 0 });
  });

  it('does not change the glossary it was given', () => {
    const current = { 道玄: 'Dao Xuan' };
    mergeGlossary(current, { 青云: 'Qingyun' });
    expect(current).toEqual({ 道玄: 'Dao Xuan' });
  });
});

describe('withLearned', () => {
  it('carries over a term the run learned while this list sat still', () => {
    expect(withLearned({ 道玄: 'Dao Xuan' }, { 道玄: 'Dao Xuan' }, { 道玄: 'Dao Xuan', 青云: 'Qingyun' }))
      .toEqual({ 道玄: 'Dao Xuan', 青云: 'Qingyun' });
  });

  it('leaves a deleted term deleted', () => {
    // It was in `saved`, so it was seen on screen and removed on purpose.
    expect(withLearned({}, { 道玄: 'Dao Xuan' }, { 道玄: 'Dao Xuan' })).toEqual({});
  });

  it('lets an edit on screen win over the copy on the server', () => {
    expect(withLearned({ 道玄: 'Daoxuan' }, { 道玄: 'Dao Xuan' }, { 道玄: 'Dao Xuan' })).toEqual({
      道玄: 'Daoxuan',
    });
  });
});

describe('planImport', () => {
  it('merges against terms the run has learned, not just the rows on screen', () => {
    // The run settled on Ivan; this editor was opened before that happened.
    const plan = planImport('{"Иван": "John"}', {}, {}, { Иван: 'Ivan' });

    expect(plan.terms).toEqual({ Иван: 'Ivan' });
    expect(plan).toMatchObject({ added: 0, kept: 1, differed: 1 });
  });

  it('imports normally when the run has learned nothing', () => {
    const plan = planImport('{"Иван": "Ivan"}', {}, {}, {});
    expect(plan.terms).toEqual({ Иван: 'Ivan' });
    expect(plan.added).toBe(1);
  });

  it('passes a bad file’s complaint straight through', () => {
    expect(() => planImport('nonsense', {}, {}, {})).toThrow(/isn’t JSON/);
  });
});

describe('describeImport', () => {
  it('counts what came in', () => {
    expect(describeImport(report({ added: 1 }))).toBe('Imported 1 term.');
    expect(describeImport(report({ added: 4 }))).toBe('Imported 4 terms.');
  });

  it('accounts for the ones already here, singular and plural', () => {
    expect(describeImport(report({ added: 4, kept: 1 }))).toBe(
      'Imported 4 terms. 1 was already here.',
    );
    expect(describeImport(report({ added: 4, kept: 2 }))).toBe(
      'Imported 4 terms. 2 were already here.',
    );
  });

  it('says when the file disagreed and lost', () => {
    expect(describeImport(report({ added: 3, kept: 12, differed: 2 }))).toBe(
      'Imported 3 terms. 12 were already here. 2 of those were translated differently in the ' +
        'file, and this book’s own was kept.',
    );
    expect(describeImport(report({ added: 3, kept: 1, differed: 1 }))).toBe(
      'Imported 3 terms. 1 was already here. One of those was translated differently in the ' +
        'file, and this book’s own was kept.',
    );
  });

  it('reports a file that brought nothing new', () => {
    expect(describeImport(report({ kept: 1 }))).toBe('Nothing new — that term was already here.');
    expect(describeImport(report({ kept: 5 }))).toBe('Nothing new — all 5 terms were already here.');
  });

  it('still explains itself when nothing was new and the file disagreed', () => {
    expect(describeImport(report({ kept: 5, differed: 1 }))).toBe(
      'Nothing new — all 5 terms were already here. One of those was translated differently in ' +
        'the file, and this book’s own was kept.',
    );
  });

  it('has something to say about an empty file', () => {
    expect(describeImport(report({}))).toBe('That file had no terms in it.');
  });
});
