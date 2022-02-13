%%
function new_Field = HandleSingularity(Field)

new_Field = Field;

indlist = find(isnan(Field)==1);
new_Field(indlist) = 0;

indlist2 = find(isinf(abs(Field))==1);
new_Field(indlist2) = 0;